import torch

# Randomly generate a set of parameters for the camera

def rand_pitch(range=26):
    return 2 * (torch.rand(1) - 0.5) * (range / 180 * torch.pi) + torch.pi/2

def rand_yaw(range=49):
    return 2 * (torch.rand(1) - 0.5) * (range / 180 * torch.pi) + torch.pi/2

def rand_cx(range=0.2):
    return 2 * (torch.rand(1) - 0.5) * range + 0.5

def rand_cy(range=0.2):
    return 2 * (torch.rand(1) - 0.5) * range + 0.5

def rand_fov(range=4.8):
    return 2 * (torch.rand(1) - 0.5) * range + 18.837




