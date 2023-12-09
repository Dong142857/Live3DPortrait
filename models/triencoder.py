import sys
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))
sys.path.append(".")
sys.path.append("..")
import math
import torch
from torch import nn
from configs.paths_config import model_paths
from PIL import Image
import numpy as np
from torch_utils import misc
from models.model import EG3DInvEncoder
from models.model import TriGenerator, TriplaneRenderer
from configs import paths_config
from models.eg3d.triplane import TriPlaneGenerator
from models.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from models.eg3d.dual_discriminator import DualDiscriminator

class TriEncoder(nn.Module):
    def __init__(self):
        super(TriEncoder, self).__init__()
        self.device = "cuda"
        self.set_encoder()
        self.set_eg3d()

    def set_eg3d(self):
        rendering_options = {
            'image_resolution': 128,
            'disparity_space_sampling': False,
            'clamp_mode': 'softplus',
            'sr_antialias': True,
        }

        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 2.25, # near point along each ray to start taking samples.
            'ray_end': 3.3, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })

        img_resolution, img_channels = 512, 3
        
        with open(paths_config.model_paths["eg3d_rebalanced"], 'rb') as f:
            resume_data = torch.load(f)
            self.G = TriPlaneGenerator(*resume_data["G_init_args"], **resume_data["G_init_kwargs"]).eval().requires_grad_(False).to(self.device)
            self.G.load_state_dict(resume_data['G'])
            self.G.neural_rendering_resolution = resume_data["G_neural_rendering_resolution"]
            self.G.rendering_kwargs = resume_data["G_renderinig_kwargs"]

            self.D = DualDiscriminator(*resume_data["D_init_args"], **resume_data["D_init_kwargs"]).eval().requires_grad_(False).to(self.device)
            self.D.load_state_dict(resume_data['D'])

        # self.triplane_generator = TriGenerator(z_dim=self.G.z_dim, c_dim=self.G.c_dim, w_dim=self.G.w_dim, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.triplane_generator = TriGenerator(*resume_data["G_init_args"], **resume_data["G_init_kwargs"]).eval().requires_grad_(False).to(self.device)
        misc.copy_params_and_buffers(self.G.backbone, self.triplane_generator.backbone, require_all=True)
        self.triplane_renderer = TriplaneRenderer(img_resolution=img_resolution, img_channels=img_channels, rendering_kwargs=rendering_options).eval().requires_grad_(False).to(self.device)
        misc.copy_params_and_buffers(self.G.decoder, self.triplane_renderer.decoder, require_all=True)
        misc.copy_params_and_buffers(self.G.superresolution, self.triplane_renderer.superresolution, require_all=True)
        # misc.copy_params_and_buffers(self.G, triplane_generator, require_all=False)
        # misc.copy_params_and_buffers(self.G, triplane_renderer, require_all=False)
        # self.triplane_renderer.requires_grad_(True)
        self.triplane_renderer.neural_rendering_resolution = self.G.neural_rendering_resolution
        self.triplane_generator.rendering_kwargs = self.G.rendering_kwargs
        self.triplane_renderer.rendering_kwargs = self.G.rendering_kwargs

    @staticmethod
    def FOV_cxy_to_intrinsics(fov_deg, cx, cy, device='cuda'):
        """Converts FOV and image center to a 3x3 camera intrinsic matrix."""
        focal_length = float(1 / (math.tan(fov_deg * 3.14159 / 360) * 1.414))
        intrinsics = torch.tensor([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], device=device)
        return intrinsics

    def eural_to_camera(self, batch_size, pitch, yaw, fov_deg=18.837, cx=0.5, cy=0.5):
        # intrinsics = FOV_to_intrinsics(fov_deg, device=self.device).reshape(-1, 9).repeat(batch_size, 1)
        intrinsics = self.FOV_cxy_to_intrinsics(fov_deg, cx, cy, device=self.device).reshape(-1, 9).repeat(batch_size, 1)
        cam_pivot = torch.tensor(self.triplane_renderer.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]), device=self.device)
        cam_radius = self.triplane_renderer.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(pitch, yaw, cam_pivot, radius=cam_radius, batch_size=batch_size, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics], 1)
        return camera_params

    @torch.no_grad()
    def sample_triplane(self, batch_size, pitch, yaw, fov_deg=18.837, cx=0.5, cy=0.5):
        z = torch.randn((batch_size, self.G.z_dim)).to(self.device)
        truncation_psi = 1
        truncation_cutoff = 14

        camera_params = self.eural_to_camera(batch_size, pitch, yaw, fov_deg, cx, cy)
        ws = self.triplane_generator.mapping(z, camera_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        triplanes = self.triplane_generator.synthesis(ws)
        # img = self.triplane_renderer.synthesis(triplanes, camera_params)['image']
        # gt = self.G.synthesis(ws, camera_params)['image']
        gt = self.G.synthesis(ws, camera_params)
        return triplanes, gt, camera_params, ws
    
    @torch.no_grad()
    def render_from_pretrain(self, batch_size, pitch, yaw, ws, fov_deg=18.837, cx=0.5, cy=0.5):
        camera_params = self.eural_to_camera(batch_size, pitch, yaw, fov_deg, cx, cy)
        gt = self.G.synthesis(ws, camera_params)
        return gt, camera_params

    @torch.no_grad()
    def sample_from_synthesis(self, batch_size, pitch, yaw):
        z = torch.randn((batch_size, self.G.z_dim)).to(self.device)

        cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]), device=self.device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        truncation_psi = 1
        truncation_cutoff = 14
        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device).reshape(-1, 9).repeat(batch_size, 1)
        # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        cam2world_pose = LookAtPoseSampler.sample(pitch, yaw, cam_pivot, radius=cam_radius, batch_size=batch_size, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(pitch, yaw, cam_pivot, radius=cam_radius, batch_size=batch_size, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics], 1)
        ws = self.G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.G.synthesis(ws, camera_params)['image']
        # visualize for debug
        # Image.fromarray(((1 + img[0].clamp(-1,1)).cpu().numpy().transpose(1, 2, 0) / 2 * 255).astype(np.uint8)).save('test.png')
        return img

    def set_encoder(self):
        self.encoder = EG3DInvEncoder(in_channels=5, encoder_name="resnet34", encoder_depth=3, mode="Normal", use_bn=False)

    def forward(self, x, c):
        '''
            Input x: [B, 5, 512, 512]
            Output dict{'image', 'image_raw', ...}
        '''
        x = self.encoder(x) # [B, 96, 256, 256]
        x = x.contiguous()
        x = self.triplane_renderer(x, c)
        return x


def debug():
    B = 2
    encoder = TriEncoder()
    # a = torch.rand((B, 5, 512, 512))
    # c = torch.rand((B, 25))
    # b = encoder(a, c)
    # print(b['image'].shape)
    # rand_pitch = (torch.rand(1) - 0.5) * torch.pi
    # rand_yaw = (torch.rand(1) - 0.5) * torch.pi
    # print(encoder.sample_from_synthesis(B, rand_pitch.cuda(), rand_yaw.cuda()).shape)
    for i in range(100):
        # rand_pitch = (torch.rand(1) - 0.5) * torch.pi/2 + torch.pi/2
        # rand_yaw = (torch.rand(1) - 0.5) * torch.pi/2 + torch.pi/2
        rand_pitch = (torch.rand(1) - 0.5) * (26.0 / 180 * torch.pi) + torch.pi/2
        rand_yaw = (torch.rand(1) - 0.5) * (39.0 / 180 * torch.pi) + torch.pi/2
        rand_cx = (torch.rand(1) - 0.5) * 0.2 + 0.5
        rand_cy = (torch.rand(1) - 0.5) * 0.2 + 0.5
        rand_fov_deg = (torch.rand(1) - 0.5) * 4.8 + 18.837
        # img = encoder.sample_from_synthesis(B, rand_pitch.cuda(), rand_yaw.cuda())
        triplanes, gt, camera_params, ws = encoder.sample_triplane(B, rand_pitch.cuda(), rand_yaw.cuda(), rand_fov_deg.cuda(), rand_cx.cuda(), rand_cy.cuda())
        Image.fromarray(((1 + gt['image'][0].clamp(-1,1)).cpu().numpy().transpose(1, 2, 0) / 2 * 255).astype(np.uint8)).save('test' + str(i) + '.png')

if __name__ == '__main__':
    debug()
