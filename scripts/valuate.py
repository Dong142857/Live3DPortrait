import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm

import sys
sys.path.append(".")
sys.path.append("..")

from configs.paths_config import val_paths
from models.Inception.inception import InceptionV3
from models.triencoder import TriEncoder
# valuate each pretrain weight...
# self used script ...

device = 'cuda'

def gen_rand_pose(pitch_range=26, yaw_range=36, cx_range=0.2, cy_range=0.2, fov_range=4.8, mode="yaw"):
    # set range
    # pitch:  +-26, yaw: +-36/+-49
    if mode == "yaw":
        return (torch.rand(1, device=device) - 0.5) * (pitch_range / 180 * torch.pi) + torch.pi/2
    elif mode == "pitch":
        return (torch.rand(1, device=device) - 0.5) * (yaw_range / 180 * torch.pi) + torch.pi/2
    else:
        raise NotImplementedError

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

@torch.no_grad()
def compute_fid(net, model_path, batch_size, n_sample=10000):
    inception = InceptionV3([3], normalize_input=False).to(device)
    inception.eval()

    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)

    batch_sizes = [batch_size] * n_batch
    batch_sizes += [resid] if resid > 0 else []

    sample_mean = np.load(val_paths['ffhq_mean'])
    sample_cov = np.load(val_paths['ffhq_cov'])

    # ---------------------------------------------------------------------------------------------------------
    print("pred.....")
    pred_features = []
    for batch in tqdm(batch_sizes):
        # print(batch)
        _, encoder_input, camera_param, ws = net.sample_triplane(batch, gen_rand_pose(pitch_range=26, mode='pitch'), gen_rand_pose(yaw_range=49, mode='yaw'))
        gen_triplanes = net.encoder(encoder_input['image'])
        render_res = net.triplane_renderer(gen_triplanes, camera_param)
        # mul_res, _ = net.render_from_pretrain(batch, gen_rand_pose(pitch_range=26, mode='pitch'), gen_rand_pose(yaw_range=49, mode='yaw'), ws) # 
        img = render_res['image']
        # img = mul_res['image']
        feat = inception(img)[0].view(img.shape[0], -1)
        pred_features.append(feat.to("cpu"))
    pred_features = torch.cat(pred_features, 0).numpy()
    pred_sample_mean = np.mean(pred_features, 0)
    pred_sample_cov = np.cov(pred_features, rowvar=False)

    basename = os.path.basename(model_path)
    np.save(f"assets/temp/{basename}_fid_pred_sample_cov{n_sample}.npy", pred_sample_mean)
    np.save(f"assets/temp/{basename}_fid_pred_sample_cov{n_sample}.npy", pred_sample_cov)
    print("pred done")
    print("calc fid....")
    # ---------------------------------------------------------------------------------------------------------
    fid = calc_fid(pred_sample_mean, pred_sample_cov, sample_mean, sample_cov)
    print("fid:", fid)
    pass


def testfid50k(model_path):
    # load model
    model_dict = torch.load(model_path)
    net = TriEncoder().to(device)
    net.encoder.load_state_dict(model_dict['encoder_state_dict'])
    net.triplane_renderer.load_state_dict(model_dict['renderer_state_dict'])

    compute_fid(net, model_path, batch_size=16, n_sample=50000)
    del net

def main():
    ckpt_path = "exp/checkpoints"
    ckpt_list = os.listdir(ckpt_path)
    for ckpt in ckpt_list:
        if ckpt.endswith(".pt"):
            model_path = os.path.join(ckpt_path, ckpt)
            print(model_path)
            print("test fid 50k...")
            testfid50k(model_path)
    pass

if __name__ == '__main__':
    main()
    pass

