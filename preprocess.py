import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as sio
import pywavefront as pf
import random
import re
import mesh_to_sdf as mts
import trimesh

def convert_to_arrays():
    """ Converts a wavefront obj into a 3D array
    DEPRECATED FUNCTION
    """
    file_dir = '.\\square_rings\\0.obj'
    scene = pf.Wavefront(file_dir, collect_faces=True)
    for name, material in scene.materials.items():
        # note: there's ony one item in the scene
        vert_arr = np.asarray(material.vertices)
        vert_arr = np.reshape(vert_arr, (int(len(vert_arr) / 3), 3)) # float64
        vert_arr_t = np.transpose(vert_arr)
        # visualize_scatterplot(vert_arr_t)

    scene_vertices = np.asarray(scene.vertices)
    # visualize_scatterplot(np.transpose(scene_vertices))
    xmax, ymax, zmax = np.amax( scene_vertices, axis = 0 )
    xmin, ymin, zmin = np.amin( scene_vertices, axis = 0 )
    scene_box = ((xmax, ymax, zmax), (xmin, ymin, zmin))
    print(scene_box)

    vox_res = (16, 16, 16)

    vox_unit_x = (xmax - xmin) / vox_res[0]
    vox_unit_y = (ymax - ymin) / vox_res[1]
    vox_unit_z = (zmax - zmin) / vox_res[2]

    for mesh in scene.mesh_list:
        print(len(np.asarray(mesh.faces)))
        for face in mesh.faces:
            print(face)

    ## TO BE CONTINUED

def visualize_3d_arr(arr):
    """ Visualize a 3 dimensional array that represents voxels
    Args:
        arr (numpy.ndarray): a 3D array representing voxels
    """
    x = arr[0]
    y = arr[1]
    z = arr[2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(arr, edgecolor='k')
    plt.show()

def visualize_scatterplot(arr):
    """Visualize points in 3D space

    Args:
        arr (numpy.ndarray): a 2D array, where the element corresponds to a dim
    """
    x = arr[0]
    y = arr[1]
    z = arr[2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()

def voxelize_and_save():
    """Converts wavefront objects to 3D arrays representing voxels.
    """
    for filename in os.listdir('./square_rings_objs'):
        if filename.endswith('.obj'):
            name = re.search('(.*).obj', filename).group(1)
            name = name + '.npy'
            if name not in os.listdir('./square_rings_vox_64'):
                print("--- Voxelizing {} ---".format(filename))
                mesh = trimesh.load(os.path.join('square_rings_objs', filename))
                voxels = mts.mesh_to_voxels(mesh, 62, pad=True)
                voxels = np.asarray(voxels)
                voxels = np.sign(voxels)
                voxels = (-1 * voxels + 1) / 2
                voxels = voxels.astype('int')
                np.save('.\\square_rings_vox_64\\' + name, voxels)
    print("finished voxelization")

if __name__ == '__main__':
    # voxelize_and_save()

    """
    test if procedure is valid
    mesh = trimesh.load('./0.obj')
    voxels = mts.mesh_to_voxels(mesh, 62, pad=True)
    voxels = np.asarray(voxels)
    voxels = np.sign(voxels)
    voxels = (-1 * voxels + 1) / 2
    voxels = voxels.astype('int')
    np.save('./0.npy', voxels)

    voxel_model_64 = np.load('./0.npy')
    voxel_model_64 = voxel_model_64.astype('int')
    visualize_3d_arr(voxel_model_64)
    """


