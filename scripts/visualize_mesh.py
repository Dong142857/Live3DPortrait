import os
import math
import trimesh
import numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender # requires Python 3.8

from matplotlib import pyplot as plt
# from load_blender import pose_spherical
'''
    This code is used to visualize the mesh of the 3D model.
'''
# 使用pyrender进行可视化ply文件
#
# import trimesh
# import pyrender
# import numpy as np
#
# mesh = trimesh.load('output/mesh_1.ply')
# scene = pyrender.Scene()
# scene.add(pyrender.Mesh.from_trimesh(mesh))
# pyrender.Viewer(scene, use_raymond_lighting=True)


def normalize_vecs(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-5)

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    print(forward_vector.shape)
    # up_vector = np.array([0.0, 1.0, 0.0]).expand_as(forward_vector)
    up_vector = np.array([0.0, 1.0, 0.0])

    right_vector = -normalize_vecs(np.cross(up_vector, forward_vector, axis=-1))
    up_vector = normalize_vecs(np.cross(forward_vector, right_vector, axis=-1))

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :]
    assert(cam2world.shape == (4, 4))
    return cam2world


def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
    h = horizontal_mean
    v = vertical_mean
    v = np.clip(v, 1e-5, math.pi - 1e-5)

    theta = h
    v = v / math.pi
    phi = np.arccos(1 - 2*v)

    camera_origins = np.zeros((batch_size, 3))

    camera_origins[:, 0:1] = radius*np.sin(phi) * np.cos(math.pi-theta)
    camera_origins[:, 2:3] = radius*np.sin(phi) * np.sin(math.pi-theta)
    camera_origins[:, 1:2] = radius*np.cos(phi)

    # forward_vectors = math_utils.normalize_vecs(-camera_origins)
    forward_vectors = normalize_vecs(lookat_position - camera_origins)
    return create_cam2world_matrix(forward_vectors, camera_origins)

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    c2w = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) @ c2w
    return c2w

# camera_params = np.concatenate([cam2world_pose.reshape(-1, 16), intrinsics], 1)

def plot_3d(mesh):
    scene = pyrender.Scene()
    # set background color
    # scene.bg_color = [0.0, 0.0, 0.0, 0.0]


    # scene.add(pyrender.Mesh.from_trimesh(mesh))
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    # camera = pyrender.PerspectiveCamera(yfov=18.834 * np.pi / 180.0, aspectRatio=1.0)
    camera = pyrender.PerspectiveCamera(yfov=1.5 * 18.834 * np.pi / 180.0, aspectRatio=1, znear=0.1)
    # camera = pyrender.OrthographicCamera(xmag=1, ymag=1)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

    # mesh = pyrender.Mesh.from_trimesh(mesh)
    # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
    #             innerConeAngle=np.pi/4)
    # r = pyrender.OffscreenRenderer(512, 512)
    # camera = pyrender.OrthographicCamera(xmag=0.6, ymag=0.6)

    camera_pose = pose_spherical(90., 0., 2.7)
    # cam_pivot = [0, 0, 0]
    # cam_radius = 2.7
    # cam2world_pose = sample(0, np.pi/2, cam_pivot, radius=cam_radius)
    # camera_pose = cam2world_pose
    # # camera_pose[:, 0] = -camera_pose[:, 0]
    # # camera_pose[0, 0] = -camera_pose[0, 0]
    # camera_pose[:, 2] = -camera_pose[:, 2]
    # camera_pose[:, 1] = -camera_pose[:, 1]
    # # camera_pose[2, :] = -camera_pose[2, :]
    # # camera_pose[1, :] = -camera_pose[1, :]
    # print(camera_pose)
    # # camera_pose[:3, 3] = 512 * camera_pose[:3, 3]
    nc = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(nc)
    scene.add(camera, pose=camera_pose)

    # Set up the light -- a point light in the same spot as the camera
    # use double sided lighting

    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/4.0)
    # light = pyrender.PointLight(color=np.ones(3), intensity=4.0)

    nl = pyrender.Node(light=light, matrix=camera_pose)
    scene.add_node(nl)
    # scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, depth = r.render(scene)

    # save the image
    # print(color.min())
    # print(depth.min(), depth.max())
    # color = ((color - color.min())/ (color.max() - color.min()) * 255.0).astype(np.uint8)
    # depth = (depth - depth.min())/ (depth.max() - depth.min()) * 255.0
    plt.imsave("output/mesh_1.png", color)
    plt.imsave("output/mesh_1_depth.png", (depth).astype(np.uint8))




if __name__ == '__main__':

    # mesh_path = "/raid/xjd/workspace/eg3d/eg3d/out/seed0000.ply"
    mesh_path = "/raid/xjd/workspace/opensource/Live3DPortrait/output/mesh_1.ply"
    # mesh_path = "/raid/xjd/workspace/other/Deep3DFaceRecon_pytorch/checkpoints/face_recon/results/epoch_20_000000/000028.obj"
    mesh = trimesh.load(mesh_path)

    # b_min_np = np.array([-bound, -bound, -bound])
    # b_max_np = np.array([ bound,  bound,  bound])
    
    # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    # return vertices.astype('float32'), faces
    # mesh.vertices = mesh.vertices.astype('float64')
    # mesh.vertices = 2.0 * (mesh.vertices / 512.0) - 1.0
    # print(mesh.vertices.dtype)
    # mesh.vertices = (mesh.vertices - (256.0, 256.0, 256.0))/256.0
    print(mesh.vertices.min())
    print(mesh.vertices.max())
    # mesh.vertices = mesh.vertices / 512.0
    # print(mesh.vertices)
    # print(mesh.vertices.shape)
    # print(mesh.faces.shape)
    plot_3d(mesh)

    

    
