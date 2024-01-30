import os
import sys
sys.path.append('.')
sys.path.append('..')
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
from configs.paths_config import model_paths
from models.triencoder import TriEncoder
from models.deep3d_module import create_model
from models.deep3d_module.render import MeshRenderer
from scripts.preprocess_single_image import align_img, lm3d_std

def compute_rotation(angles, device):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)

def process_image(image):
    # image = Image.open(image_path).convert('RGB')
    mtcnn = MTCNN(keep_all=True)

    # Detect face using MTCNN
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    _,H = image.size

    lm = points
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    _, im_high, _, _, = align_img(image, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)

    left = int(im_high.size[0]/2 - center_crop_size/2)
    upper = int(im_high.size[1]/2 - center_crop_size/2)
    right = left + center_crop_size
    lower = upper + center_crop_size
    im_cropped = im_high.crop((left, upper, right,lower))
    im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
    return im_cropped



class pose_estimator:
    def __init__(self, device, test=False):
        self.device = device
        self.deep3D = create_model()
        self.deep3D.setup()
        self.deep3D.device = device
        if not test:
            self.deep3D.parallelize()
        self.deep3D.eval()
        self.deep3D.facemodel.to(device)
        self.deep3D.net_recon.to(device)

    def preprocess(self, img):
        return F.interpolate((img + 1) / 2.0, size = (224, 224), mode = 'bilinear', align_corners = False)

    def estimate_pose(self, img):
        batch_size = img.shape[0] # 1
        # 当前只能处理一张图像
        im_tensor = self.preprocess(img)
        output_coeff = self.deep3D.net_recon(im_tensor)
        angle = output_coeff[:, 224: 227]
        trans = output_coeff[:, 254: 257]
        R = compute_rotation(angle, device=self.device)
        R = R[0].detach().cpu().numpy()
        trans = trans[0].detach().cpu().numpy()
        angle = angle[0].detach().cpu().numpy()
        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R

        c *= 0.27 # normalize camera radius
        c[1] += 0.006 # additional offset used in submission
        c[2] += 0.161 # additional offset used in submission
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2]

        focal = 2985.29 # = 1015*1024/224*(300/466.285)#
        pp = 512#112
        w = 1024#224
        h = 1024#224

        count = 0
        K = np.eye(3)
        K[0][0] = 4.26
        K[1][1] = 4.26
        K[0][2] = 0.5
        K[1][2] = 0.5

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        camera_p = torch.from_numpy(np.concatenate([pose.flatten(), K.flatten()], dtype=np.float32)).unsqueeze(0).to(self.device) 
        return camera_p
    

'''
    1 load the model, and pose estimator
    2 load the video
    3 for each frame, estimate the pose
    4 run the encoder and renderer 
    5 save the video
'''
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load model and pose estimator
    pose_est = pose_estimator(device, test=True)
    model_path = model_paths["encoder_render"]
    mode = 'Normal'
    model_dict = torch.load(model_path)
    net = TriEncoder(mode=mode)
    net.encoder.load_state_dict(model_dict['encoder_state_dict'])
    net.triplane_renderer.load_state_dict(model_dict['renderer_state_dict'])

    net.encoder.eval()
    net.triplane_renderer.eval()
    net.encoder.to(device)
    net.triplane_renderer.to(device)

    # load video and process, align and crop , predict 3D pose
    video_path = '/raid/xjd/workspace/opensource/Live3DPortrait/output/video_1.mp4'
    cap = cv2.VideoCapture(video_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # image_list = []
    video_kwargs = {}
    video_out = imageio.get_writer("output.mp4", mode='I', fps=60, codec='libx264', **video_kwargs)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        process_frame = process_image(source)
        frame = transform(process_frame).unsqueeze(0).to(device)
        img_tensor = frame * 2 - 1
        camera_p = pose_est.estimate_pose(frame)
        # print(camera_p)
        # run the encoder and renderer
        with torch.no_grad():
            gen_triplanes = net.encoder(img_tensor)
            # net_camera_p = net.eural_to_camera(1, np.pi/2, np.pi/2) # pitch yawl
            # print(net_camera_p)
            render_res = net.triplane_renderer(gen_triplanes, camera_p)
            img = (1 + render_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
        # Image.fromarray(np.concatenate((np.array(process_frame), img), axis=1).astype(np.uint8)).save('output.png')
        # image_list.append(np.concatenate((np.array(process_frame), img), axis=1))
        video_out.append_data(np.concatenate((np.array(process_frame), img), axis=1).astype(np.uint8))
        # break
    # save the video
    video_out.close()
    cap.release()


if __name__ == '__main__':
    main()
    pass

