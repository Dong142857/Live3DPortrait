import sys
sys.path.append('./')
sys.path.append('../')

import os
import numpy as np
import torch
import imageio
import argparse
import trimesh
import pyrender
import mcubes
import json
from tqdm import tqdm

# from training.utils import color_mask, color_list
from models.triencoder import TriEncoder
from configs.paths_config import model_paths

os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_sigma_field_np(nerf, styles, resolution=512, block_resolution=64):
    # return numpy array of forwarded sigma value
    # bound = (nerf.rendering_kwargs['ray_end'] - nerf.rendering_kwargs['ray_start']) * 0.5
    bound = nerf.rendering_kwargs['box_warp'] * 0.5
    X = torch.linspace(-bound, bound, resolution).split(block_resolution)

    sigma_np = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(X):
            for zi, zs in enumerate(X):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).to(styles.device)  # B, H, H, H, C
                block_shape = [1, len(xs), len(ys), len(zs)]
                out = nerf.sample_mixed(pts.reshape(1,-1,3), None, ws=styles, noise_mode='const')
                feat_out, sigma_out = out['rgb'], out['sigma']
                sigma_np[xi * block_resolution: xi * block_resolution + len(xs), \
                yi * block_resolution: yi * block_resolution + len(ys), \
                zi * block_resolution: zi * block_resolution + len(zs)] = sigma_out.reshape(block_shape[1:]).detach().cpu().numpy()
                # print(feat_out.shape)

    return sigma_np, bound


def extract_geometry(nerf, styles, resolution, threshold):

    # print('threshold: {}'.format(threshold))
    u, bound = get_sigma_field_np(nerf, styles, resolution)
    vertices, faces = mcubes.marching_cubes(u, threshold)
    # vertices, faces, normals, values = skimage.measure.marching_cubes(
    #     u, level=10
    # )
    b_min_np = np.array([-bound, -bound, -bound])
    b_max_np = np.array([ bound,  bound,  bound])

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices.astype('float32'), faces


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model_paths["encoder_render"], help='model path')
    parser.add_argument('--image', type=str, default='./imgs/1.jpg', help='image path')
    parser.add_argument('--output', type=str, default='./output', help='output path')
    parser.add_argument('--save_mesh', type=bool, default=True, help='If true, save mesh')
    parser.add_argument('--save_video', type=bool, default=True, help='If true, save video')
    parser.add_argument('--mode', type=str, default='Normal', help='mode of triplane generation', choices=['LT', 'Normal'])
    args = parser.parse_args()


    device = 'cuda'
    model_path = args.model
    image_path = args.image
    output_path = args.output
    mode = args.mode
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # load model
    model_dict = torch.load(model_path)
    net = TriEncoder(mode=mode)
    net.encoder.load_state_dict(model_dict['encoder_state_dict'])
    net.triplane_renderer.load_state_dict(model_dict['renderer_state_dict'])



    mesh_trimesh = trimesh.Trimesh(*extract_geometry(G, ws, resolution=512, threshold=50.))

    if args.cfg == 'seg2cat' or args.cfg == 'seg2face':

        verts_np = np.array(mesh_trimesh.vertices)
        colors = torch.zeros((verts_np.shape[0], 3), device=device)
        semantic_colors = torch.zeros((verts_np.shape[0], 6), device=device)
        samples_color = torch.tensor(verts_np, device=device).unsqueeze(0).float()

        head = 0
        max_batch = 10000000
        with tqdm(total = verts_np.shape[0]) as pbar:
            with torch.no_grad():
                while head < verts_np.shape[0]:
                    torch.manual_seed(0)
                    out = G.sample_mixed(samples_color[:, head:head+max_batch], None, ws, truncation_psi=1, noise_mode='const')
                    # sigma = out['sigma']
                    colors[head:head+max_batch, :] = out['rgb'][0,:,:3]
                    seg = out['rgb'][0, :, 32:32+6]
                    semantic_colors[head:head+max_batch, :] = seg
                    # semantics[:, head:head+max_batch] = out['semantic']
                    head += max_batch
                    pbar.update(max_batch)

        semantic_colors = torch.tensor(color_list)[torch.argmax(semantic_colors, dim=-1)]

        mesh_trimesh.visual.vertex_colors = semantic_colors.cpu().numpy().astype(np.uint8)

        # Save mesh.
        mesh_trimesh.export(os.path.join(save_dir, f'semantic_mesh.ply'))
    elif args.cfg == 'edge2car':
        # Save mesh.
        mesh_trimesh.export(os.path.join(save_dir, f'{args.cfg}_mesh.ply'))

    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                innerConeAngle=np.pi/4)
    r = pyrender.OffscreenRenderer(512, 512)
    if args.cfg == 'seg2cat' or args.cfg == 'seg2face':
        camera = pyrender.OrthographicCamera(xmag=0.3, ymag=0.3)

    elif args.cfg == 'edge2car':
        camera = pyrender.OrthographicCamera(xmag=0.6, ymag=0.6)


    frames_mesh = []
    num_frames = 120

    for frame_idx in tqdm(range(num_frames)):
        scene = pyrender.Scene()
        scene.add(mesh)

        if args.cfg == 'seg2cat' or args.cfg == 'seg2face':
            camera_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / num_frames),
                                                3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / num_frames),
                                                torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device), radius=1, device=device)
        elif args.cfg == 'edge2car':
            camera_pose = LookAtPoseSampler.sample(-3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / num_frames),
                                                3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / num_frames),
                                                torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device), radius=1.2, device=device)
        camera_pose = camera_pose.reshape(4, 4).cpu().numpy().copy()
        camera_pose[:, 1] = -camera_pose[:, 1]
        camera_pose[:, 2] = -camera_pose[:, 2]

        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, depth = r.render(scene)
        frames_mesh.append(color)

    imageio.mimsave(os.path.join(save_dir, f'rendered_mesh.gif'), frames_mesh, fps=60)
    r.delete()



if __name__ == '__main__':
    main()


