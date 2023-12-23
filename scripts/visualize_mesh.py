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


# camera_params = np.concatenate([cam2world_pose.reshape(-1, 16), intrinsics], 1)

def plot_3d(mesh):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    # scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    # camera = pyrender.PerspectiveCamera(yfov=2 * 18.834 * np.pi / 180.0, aspectRatio=1.0)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)

    # camera_pose = pose_spherical(-20., -40., 1.).numpy()
    cam_pivot = [0, 0, 0]
    cam_radius = 2.7
    cam2world_pose = sample(-np.pi/2 + np.pi/4, np.pi/3, cam_pivot, radius=cam_radius)
    camera_pose = cam2world_pose
    nc = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(nc)

    # Set up the light -- a point light in the same spot as the camera
    light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
    nl = pyrender.Node(light=light, matrix=camera_pose)
    scene.add_node(nl)

    # Render the scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, depth = r.render(scene)

    # save the image
    print(color)
    plt.imsave("output/mesh_1.png", color)
    plt.imsave("output/mesh_1_depth.png", depth)



if __name__ == '__main__':

    mesh_path = "output/mesh_1.ply"
    mesh = trimesh.load(mesh_path)
    print(mesh.vertices.shape)
    print(mesh.faces.shape)
    plot_3d(mesh)

    pass
