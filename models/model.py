import torch
import torch.nn as nn
import torch.nn.functional as F
# import segmentation_models_pytorch
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3
'''
    impletement of RealTimeRF including the full model and LT model
'''
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from einops import rearrange, repeat

import sys
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..')))
sys.path.append(".")
sys.path.append("..")
from torch_utils.ops import upfirdn2d
from torch_utils.misc import assert_shape

from functools import reduce
from typing import Union
from segmentation_models_pytorch.encoders.mix_transformer import OverlapPatchEmbed, Block

from models.eg3d.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes
from models.eg3d.volumetric_rendering.ray_sampler import RaySampler
from models.eg3d.superresolution import SuperresolutionHybrid8XDC
from models.eg3d.networks_stylegan2 import FullyConnectedLayer 
from models.eg3d.triplane import OSGDecoder
from models.eg3d.networks_stylegan2 import Generator as StyleGAN2Backbone

class TriGenerator(nn.Module):
    '''
        similar to TriplaneGenerator class but lack of renderer
    '''
    def __init__(self,              # 参数表暂时不删，做占位用
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self._last_planes = None
        self.rendering_kwargs = rendering_kwargs

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        return planes

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


class TriplaneRenderer(nn.Module):
    def __init__(self, img_resolution, img_channels, rendering_kwargs={}) -> None:
        '''
        Triplane Renderer
            Generate 2D image from triplanes representation
            SuperResolution without stylecode 
            FullyConnected layer
        '''
        super(TriplaneRenderer, self).__init__()
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()

        self.w_dim = 512
        self.const = torch.nn.Parameter(torch.randn([1, 1, self.w_dim])) # 常数输入

        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.superresolution = SuperresolutionHybrid8XDC(32, img_resolution, sr_num_fp16_res=0, sr_antialias=True)
        self.rendering_kwargs = rendering_kwargs
        self.neural_rendering_resolution = 128 # 64

    def synthesis(self, planes, c, neural_rendering_resolution=None):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        
        N, _, _ = ray_origins.shape
        
        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, _ = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        # sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode='none')
        const_w_input = self.const.repeat([N, 1, 1])
        sr_image = self.superresolution(rgb_image, feature_image, const_w_input, noise_mode='none')

        return {'image': sr_image, 
                'image_raw': rgb_image, 
                'image_depth': depth_image, 
                'feature_image': feature_image, 
                'planes': planes}

    def sample_density(self, planes, coordinates, directions):
        '''
            给定triplanes和camera参数，生成图像并且返回
        '''
        sampled_features = sample_from_planes(self.renderer.plane_axes, planes, coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])
        out = self.decoder(sampled_features, directions)
        return out

    def forward(self, planes, c):
        '''
            给定triplanes和camera参数，生成图像并且返回
        '''
        return self.synthesis(planes, c)

# coding self RTRF
class EncoderEhigh(nn.Module):
    def __init__(self, mode="Normal") -> None:
        super(EncoderEhigh, self).__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01)
        ) if self.mode == "Normal" else nn.Sequential(
            # Input is Second layer output of DeeplabV3
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),  
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            )

        self.net.apply(self._init_weights) # 仿照timm的初始化方式

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, img):
        return self.net(img)

class EncoderF(nn.Module):
    def __init__(self, mode="Normal") -> None:
        super(EncoderF, self).__init__()

        self.mode = mode
        self.blocks_num = 5 if self.mode == 'Normal' else 2

        self.patchembed = OverlapPatchEmbed(img_size=64, stride=2, in_chans=256, embed_dim=1024)
        self.blocks = nn.ModuleList([Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1) for _ in range(self.blocks_num)])

        self.upsample = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2.0), # equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``
        )

        self.net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2.0), # equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        ) if self.mode == "Normal" else nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
        )
        self.net.apply(self._init_weights) # 仿照timm的初始化方式

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        x, H, W  = self.patchembed(x)
        for i in range(self.blocks_num):
            x = self.blocks[i](x, H, W)  # [B, N, C]
        # print((H, W)) # (32, 32)
        # print(x.shape) # [2, 1024, 1024]
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        # print(x.shape) # [2, 1024, 32, 32]
        x = self.upsample(x)
        x = self.net(x)
        return x 

class EncoderFinal(nn.Module):
    def __init__(self, mode="Normal") -> None:
        super(EncoderFinal, self).__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
        ) if self.mode == "Normal" else nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
        )

        if self.mode == "Normal":
            self.pathembed = OverlapPatchEmbed(img_size=256, stride=2, in_chans=128, embed_dim=1024)
            self.block = Block(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2)
        else:
            self.pathembed = OverlapPatchEmbed(img_size=128, stride=2, in_chans=256, embed_dim=1024)
            self.block = Block(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2)
            # self.pathembed = OverlapPatchEmbed(img_size=128, stride=2, in_chans=128, embed_dim=1024)
            # self.block = Block(dim=256, num_heads=2, mlp_ratio=2, sr_ratio=2)
        
        self.pixelShuffel = nn.PixelShuffle(upscale_factor=2)

        # concat with output of encoder F
        # self.F_feature = F_feature
        self.output_net = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1), 
        ) if self.mode == "Normal" else nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),  
            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),  
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
        )
        self.net.apply(self._init_weights) # 仿照timm的初始化方式
        self.output_net.apply(self._init_weights) # 仿照timm的初始化方式``

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, F_feature): # debug
        xx = self.net(torch.cat([x, F_feature], dim=1))
        xx, H, W = self.pathembed(xx)
        xx = self.block(xx, H, W)
        xx = rearrange(xx, "b (h w) c -> b c h w", h=H, w=W)
        xx = self.pixelShuffel(xx)
        if self.mode == "Normal":
            return self.output_net(torch.cat([xx, F_feature], dim=1))
        else:
            return self.output_net(xx)

class EG3DInvEncoder(nn.Module):
    def __init__(
        self, 
        in_channels=5, 
        encoder_name="resnet34",
        encoder_depth=3,
        mode="Normal",
        use_bn=False,
    ) -> None:
        super(EG3DInvEncoder, self).__init__()
        self.mode = mode
        deeplab = DeepLabV3(in_channels=in_channels, encoder_name=encoder_name, encoder_depth=encoder_depth) # encoder_depth [3, 5]
        self.encoder = deeplab.encoder
        self.decoder = deeplab.decoder

        if not use_bn: # 很漂亮的实现方式
            encoder_name_modules = self.encoder.named_modules()
            self.set_module(encoder_name_modules, self.encoder)
            decoder_name_modules = self.decoder.named_modules() # 对于decoder也这样操作
            self.set_module(decoder_name_modules, self.decoder) # 当ASPP中不使用可分离卷积时就不会有bug

        self.F_encoder = EncoderF(mode=self.mode)
        self.EHigh_encoder = EncoderEhigh(mode=self.mode)
        self.Final_encoder = EncoderFinal(mode=self.mode)

    def set_module(self, name_modules, root_module):
        '''
            Remove BN, Conv add bias
            Original: Conv2d + BN + ReLU (Do not use separate Conv)
            Now: Conv2d(add bias) + ReLU
        '''
        for name, m in name_modules:
            if isinstance(m, nn.BatchNorm2d): # 反射机制
                names = name.split(sep='.')
                parents = reduce(getattr, names[:-1], root_module)
                setattr(parents, names[-1], nn.Identity())
            elif isinstance(m, nn.Conv2d):
                if m.bias is not None: # may not enter here
                    m.bias.data.zero_()
                else:
                    m.bias = nn.Parameter(torch.zeros(m.out_channels))

    # optimizer utils
    def get_params(self, lr):
        # 可分训练阶段进行调整
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
            {'params': self.F_encoder.parameters(), 'lr': lr},
            {'params': self.EHigh_encoder.parameters(), 'lr': lr},
            {'params': self.Final_encoder.parameters(), 'lr': lr}, 
        ]
        return params

    def forward(self, x):
        # x is [B, 5, H, W]
        chan_num = x.shape[1]
        device = x.device
        if chan_num == 3:
            # use coordnate as the fourth and fifth channle 
            B = x.shape[0]
            H, W = x.shape[2], x.shape[3]
            grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid_x = repeat(grid_x, "h w -> b c h w", b=B, c=1).to(device)
            grid_y = repeat(grid_y, "h w -> b c h w", b=B, c=1).to(device)
            # print(x.shape, grid_x.shape, grid_y.shape)
            x = torch.cat([x, grid_x, grid_y], dim=1)
            chan_num = x.shape[1]

        assert chan_num == 5, "x.shape[1] shoule be 5"
        x_encode = self.encoder(x)
        low_feature = self.decoder(x_encode[-1])
        # low_feature = self.decoder(*x_encode)
        f_feature = self.F_encoder(low_feature)
        high_feature = self.EHigh_encoder(x)
        # if self.mode == "Normal":
        #     high_feature = self.EHigh_encoder(x)
        # elif self.mode == "LT":
        #     high_feature = self.EHigh_encoder(x_encode[1])
        # else:
        #     raise ValueError("mode should be Normal or LT")
        output = self.Final_encoder(high_feature, f_feature)


        return output


def config_option():
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

    return rendering_options

def debug_test():
    # mode = "Normal"
    # # mode = "LT"
    # a = torch.rand((2, 5, 512, 512))
    # layer = DeepLabV3(in_channels=5, encoder_name="resnet34", encoder_depth=3) # encoder_depth [3, 5]
    # encoder = layer.encoder # 先尝试，后面再改
    # decoder = layer.decoder
    # x = decoder(encoder(a)[-1])
    # print(x.shape) # torch.size(2, 256, 64, 64)
    # F = EncoderF(mode=mode)
    # f_feature = F(x)
    # print(f_feature.shape) # [2, 96, 256, 256] # LT: [2, 128, 128, 128]
    # EHigh = EncoderEhigh(mode=mode)
    # y = EHigh(a)
    # print(y.shape) # [2, 96, 256, 256] # LT: [2, 128, 128, 128]
    # # 融合
    # final_layer = EncoderFinal(mode=mode)
    # feature = final_layer(y, f_feature)
    # print(feature.shape) # [2, 96, 256, 256]
    ############################
    device = 'cuda'
    render_options = config_option()
    B = 1
    # plane = torch.rand((B, 96, 256, 256)) # 可以修改为其他的分辨率
    a = torch.rand((B, 5, 512, 512)).to(device)
    m = EG3DInvEncoder(in_channels=5, encoder_name="resnet34", encoder_depth=3, mode="LT", use_bn=False).to(device)

    m.eval()
    for i in range(50):
        a = torch.rand((B, 5, 512, 512)).to(device)
        s1 = time.time()
        plane = m(a).contiguous()
        s2 = time.time()
        print("time: {} ms".format((s2 - s1 ) * 1000))
    print(plane.shape)
    # # print(plane.shape)
    # c = [0.8452497124671936, 0.028885385021567345, -0.5335903167724609, 1.3744339756005164, -0.03056236356496811, -0.9942903518676758, -0.10223815590143204, 0.27722038701208845, -0.5334968566894531, 0.10272454470396042, -0.8395407795906067, 2.307396824072493, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]
    # c = torch.tensor(c).repeat((B, 1))
    # print(c.shape)
    
    print("Rendering... ")
    render = TriplaneRenderer(img_resolution=512, img_channels=3, rendering_kwargs=render_options).to(device)

    for i in range(50):
        c = torch.rand((B, 25)).to(device)
        s1 = time.time()
        d = render(plane, c)
        s2 = time.time()
        print("time: {} ms".format((s2 - s1 ) * 1000))

    # print(list(m.named_parameters()))
    # print(list(m.named_modules()))
    # print(d['image'].shape)
    # print(d['image_raw'].shape)
    # print(d['image_depth'].shape)
    # print(d['feature_image'].shape)
    # print(d['planes'].shape)

if __name__ == '__main__':
    debug_test()
