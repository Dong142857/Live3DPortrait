import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(".")
sys.path.append("..")
import time
import argparse

import cv2
import scipy
import torch
import numpy as np
import face_alignment as FAN
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision import transforms

from models.deep3d_module.util.preprocess import align_img
from models.deep3d_module.util.load_mats import load_lm3d
device = 'cuda'
BFM_PATH = '/raid/xjd/workspace/opensource/Live3DPortrait/models/deep3d_module/BFM'
mtcnn = MTCNN(keep_all=True, device=device)
fa = FAN.FaceAlignment(FAN.LandmarksType.TWO_D, flip_input=False)
lm3d_std = load_lm3d(BFM_PATH) 

'''
    This script is used to preprocess a video
'''

 # convert 68 landmarks to 5 landmarks
def extract_5p(lm):                                                                
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1                            
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
            lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)    
    lm5p = lm5p[[1, 2, 0, 3, 4], :]                                                
    return lm5p 

def extract_images(path, out_path, fps=25):
    # with same resolution
    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -s 512x512 -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def align(img, landmarks68, output_size=1500):
    enable_padding = True
    # transform_size = int(512 * 4096 / 1500)
    transform_size = 4096
    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = np.array(landmarks68)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    q_scale = 1.8
    x = q_scale * x
    y = q_scale * y
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    import time

    # Shrink.
    # start_time = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # print("shrink--- %s seconds ---" % (time.time() - start_time))

    # Crop.
    # start_time = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    # print("crop--- %s seconds ---" % (time.time() - start_time))

    # Pad.
    # start_time = time.time()
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        low_res = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
        blur = qsize * 0.02*0.1
        low_res = scipy.ndimage.gaussian_filter(low_res, [blur, blur, 0])
        low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LANCZOS4)
        img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
        median = np.median(median, axis=(0,1))
        img += (median - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # print("pad--- %s seconds ---" % (time.time() - start_time))

    # Transform.
    # start_time = time.time()
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)
    # print("transform--- %s seconds ---" % (time.time() - start_time))
    return img


def preprocess_single_image(image_path, output_path, eye_align=False):
    image = Image.open(image_path).convert('RGB')
    if eye_align:
        # Detect face using face_alignment
        landmarks = fa.get_landmarks_from_image(image_path)
        img = align(image, landmarks[0])
        # center crop 
        transform = transforms.Compose([
            transforms.CenterCrop(1024), # 700, FFHQ in eg3d process need their own crop param
            # transforms.Resize(512),
        ])
        image = transform(img)      
    
    # Detect face using MTCNN
    # s = time.time()
    # image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device=device)
    boxes, probs, points = mtcnn.detect(image, landmarks=True) # about 2~3.5s
    # print(f'[INFO] ===== MTCNN detect time: {time.time() - s} =====')
    _,H = image.size

    lm = points
    if boxes is None or boxes[0] is None or len(boxes) == 0:
        print(f'[INFO] ===== MTCNN detect failed =====')
        return

    lm = lm.reshape([-1, 2])
    lm = lm[:5, :] # only use 5 landmarks,select the largest prob face
    lm[:, -1] = H - 1 - lm[:, -1]


    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    # s = time.time()
    _, im_high, _, _, = align_img(image, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor) # about 0.02s
    # print(f'[INFO] ===== align_img time: {time.time() - s} =====')

    left = int(im_high.size[0]/2 - center_crop_size/2)
    upper = int(im_high.size[1]/2 - center_crop_size/2)
    right = left + center_crop_size
    lower = upper + center_crop_size
    im_cropped = im_high.crop((left, upper, right,lower))
    im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)

    basename = os.path.basename(image_path)
    im_cropped.save(os.path.join(output_path, basename))

def preprocess_video(video_path, output_path):
    ori_imgs_dir = os.path.join(output_path, 'ori_imgs')
    # two mode: crop and crop_align
    crop_imgs_dir = os.path.join(output_path, 'crop_imgs') 
    crop_align_imgs_dir = os.path.join(output_path, 'crop_align_imgs')
    if not os.path.exists(ori_imgs_dir):
        os.makedirs(ori_imgs_dir)
    if not os.path.exists(crop_imgs_dir):
        os.makedirs(crop_imgs_dir)
    if not os.path.exists(crop_align_imgs_dir):
        os.makedirs(crop_align_imgs_dir)

    # extract frames from video
    extract_images(video_path, ori_imgs_dir)
        
    # preprocess images
    for img_name in tqdm(os.listdir(ori_imgs_dir)):
        img_path = os.path.join(ori_imgs_dir, img_name)
        preprocess_single_image(img_path, crop_imgs_dir, eye_align=False)
        # preprocess_single_image(img_path, crop_align_imgs_dir, eye_align=True)


def main():
    parser = argparse.ArgumentParser(description='Preprocess video.')
    parser.add_argument('--dataset_path', type=str, default="/NAS/xjd/datasets/CelebVHQ/35666/",
                         help='Path to the image to be preprocessed.')
    parser.add_argument('--output_path', type=str, default="/NAS/xjd/datasets/CelebVHQ/preprocess", help='Path to save the preprocessed image.')
    args = parser.parse_args()

    for video_name in os.listdir(args.dataset_path):
        video_path = os.path.join(args.dataset_path, video_name) 
        output_path = os.path.join(args.output_path, video_name[:-4]) # remove .mp4
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            preprocess_video(video_path, output_path)

if __name__ == "__main__":
    main()
