import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

import imageio
import os
from tqdm import tqdm
from configs.paths_config import model_paths
from models.triencoder import TriEncoder
from models.eg3d.shape_utils import convert_sdf_samples_to_ply

device = "cuda:0"

def gen_rand_pose(device):
    return (torch.rand(1, device=device) - 0.5) * torch.pi/6 + torch.pi/2

def layout_grid(img: np.ndarray, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def generate_video(net, gen_triplanes, savepath, batch_size=1):
    voxel_resolution = 512
    video_kwargs = {}
    video_out = imageio.get_writer(savepath, mode='I', fps=60, codec='libx264', **video_kwargs)
    grid_h, grid_w = 1, 1
    num_keyframes, w_frames = 12, 24

    image_mode = "image"
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                camera_p = net.eural_to_camera(batch_size, 
                        3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                        )
                render_res = net.triplane_renderer(gen_triplanes, camera_p)
                img = render_res[image_mode][0]
                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_h=1, grid_w=1))
    video_out.close()

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def extract_mesh(net, gen_triplanes, savepath):
    shape_res =  512
    max_batch=1000000
    my_plane = gen_triplanes.reshape(1,3,32,256,256)
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=net.triplane_generator.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = net.triplane_renderer.sample_density(my_plane, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head])
                sigmas[:, head:head+max_batch] = sigma['sigma']
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, savepath, level=10)

def multi_view_output(net, gen_triplanes, savepath, batch_size=1):
    # observe in multiple views
    image_mode = "image" # or "depth"
    offset = np.linspace(-np.pi/6, np.pi/6, 8)
    imgs_list = []
    for p in offset:
        camera_p = net.eural_to_camera(batch_size, np.pi/2 + p, np.pi/2)
        render_res = net.triplane_renderer(gen_triplanes, camera_p)
        img = render_res[image_mode][0]
        imgs_list.append(img.clamp(-1,1).squeeze(0).cpu().detach().permute((1,2,0)).numpy())
    image_list = (np.concatenate(imgs_list, axis=1) + 1) / 2 * 255
    Image.fromarray(image_list.astype(np.uint8)).save(savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model_paths["encoder_render"], help='model path')
    parser.add_argument('--image', type=str, default='./imgs/1.jpg', help='image path')
    parser.add_argument('--output', type=str, default='./output', help='output path')
    parser.add_argument('--save_mesh', type=bool, default=True, help='If true, save mesh')
    parser.add_argument('--save_video', type=bool, default=True, help='If true, save video')
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    output_path = args.output
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # load model
    model_dict = torch.load(model_path)
    net = TriEncoder()
    net.encoder.load_state_dict(model_dict['encoder_state_dict'])
    net.triplane_renderer.load_state_dict(model_dict['renderer_state_dict'])

    # load image
    img = Image.open(image_path).resize((512, 512))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    if img_tensor.shape[1] == 4: # remove alpha channel
        img_tensor = img_tensor[:, :3, :, :]    
    img_tensor = img_tensor * 2 - 1

    # inference
    net.encoder.to(device)
    net.encoder.eval()
    basename = os.path.basename(image_path).split('.')[0]
    with torch.no_grad():
        gen_triplanes = net.encoder(img_tensor)
        multi_view_output(net, gen_triplanes, os.path.join(output_path, 'multi_view_' + basename + '.png'))
        if args.save_video:
            generate_video(net, gen_triplanes, os.path.join(output_path, 'video_' + basename + '.mp4'))
        if args.save_mesh:
            extract_mesh(net, gen_triplanes, os.path.join(output_path, 'mesh_' + basename + '.ply'))






