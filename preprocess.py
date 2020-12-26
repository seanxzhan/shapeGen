import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as sio
import pywavefront as pf

def check_data():
    """
    Checks if data can be loaded by scipy.io
    """
    all_mat = glob.glob('.\\02691156\\*.mat')
    # model = sio.loadmat('.\\02691156\\1a6ad7a24bb89733f412783097373bdc.mat')
    for f in all_mat:
        try:
            model = sio.loadmat(f)
        except:
            print("err loading: " + f)
    print("------ FINISHED CHECKING ------")

def convert_to_arrays():
    file_dir = '.\\square_rings\\0.obj'
    scene = pf.Wavefront(file_dir)
    print(scene)

def point_sampling():
    voxel_model_mat = sio.loadmat('.\\02691156\\1a6ad7a24bb89733f412783097373bdc.mat')
    voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
    voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
    visualize_3d_arr(voxel_model_bi)
    print("--- voxel_model_b ---")
    print(voxel_model_b)
    print("--- voxel_model_bi ---")
    print(voxel_model_bi)
    
    voxel_model_256 = np.zeros([256,256,256],np.uint8)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = \
                    voxel_model_b[voxel_model_bi[i,j,k]]
    print("--- voxel_model_256 ---")
    print(voxel_model_256)
    # visualize_3d_arr(voxel_model_256)

def visualize_3d_arr(arr):
    x = arr[0]
    y = arr[1]
    z = arr[2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(arr, edgecolor='k')
    plt.show()

if __name__ == '__main__':
    # check_data()
    # convert_to_arrays()
    # model = sio.loadmat('.\\02691156\\1a6ad7a24bb89733f412783097373bdc.mat')
    # print(model)
    point_sampling()