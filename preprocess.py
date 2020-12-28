import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as sio
import pywavefront as pf
import random

dim = 64

vox_size_1 = 16
vox_size_2 = 32
vox_size_3 = 64

batch_size_1 = 16*16*16
batch_size_2 = 16*16*16
batch_size_3 = 16*16*16*4

def check_data():
    """ Checks if data can be loaded by scipy.io
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
    """ Converts a wavefront obj into a 3D array
    """
    file_dir = '.\\square_rings\\4.obj'
    scene = pf.Wavefront(file_dir, collect_faces=True)
    for name, material in scene.materials.items():
        # note: there's ony one item in the scene
        vert_arr = np.asarray(material.vertices)
        vert_arr = np.reshape(vert_arr, (int(len(vert_arr) / 3), 3)) # float64
        vert_arr_t = np.transpose(vert_arr)
        # visualize_scatterplot(vert_arr_t)

    scene_vertices = np.asarray(scene.vertices)
    # visualize_scatterplot(np.transpose(scene_vertices))
    scene_box = (scene.vertices[0], scene.vertices[0])
    for vertex in scene.vertices:
        min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
        max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
        scene_box = (min_v, max_v)
    # print(scene_box)

    print(len(scene.meshes))
    print(scene.mesh_list[0])

    # for mesh in scene.mesh_list:
    #     for face in mesh.faces:
    #         print(face)

def sample_point_in_cube(block,target_value,halfie):
	halfie2 = halfie*2
	
	for i in range(100):
		x = np.random.randint(halfie2)
		y = np.random.randint(halfie2)
		z = np.random.randint(halfie2)
		if block[x,y,z]==target_value:
			return x,y,z
	
	if block[halfie,halfie,halfie]==target_value:
		return halfie,halfie,halfie
	
	i=1
	ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
	if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
		return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	
	for i in range(2,halfie+1):
		six = [(halfie-i,halfie,halfie),(halfie+i-1,halfie,halfie),(halfie,halfie,halfie-i),(halfie,halfie,halfie+i-1),(halfie,halfie-i,halfie),(halfie,halfie+i-1,halfie)]
		for j in range(6):
			if block[six[j]]==target_value:
				return six[j]
		ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
		if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
			return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	print('hey, error in your code!')
	exit(0)

def IMNET_point_sampling():
    voxel_model_mat = sio.loadmat('.\\02691156\\1a6ad7a24bb89733f412783097373bdc.mat')
    # voxel_model_mat = sio.loadmat('.\\02691156\\1a9b552befd6306cc8f2d5fe7449af61.mat')
    voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
    voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
    visualize_3d_arr(voxel_model_bi)
    
    voxel_model_256 = np.zeros([256,256,256],np.uint8)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = \
                    voxel_model_b[voxel_model_bi[i,j,k]]
    voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)

    #carve the voxels from side views:
    #top direction = Y(j) positive direction
    dim_voxel = 256
    top_view = np.max(voxel_model_256, axis=1)
    left_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
    left_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
    front_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
    front_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
    
    for j in range(dim_voxel):
        for k in range(dim_voxel):
            occupied = False
            for i in range(dim_voxel):
                if voxel_model_256[i,j,k]>0:
                    if not occupied:
                        occupied = True
                        left_min[j,k] = i
                    left_max[j,k] = i
    
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            occupied = False
            for k in range(dim_voxel):
                if voxel_model_256[i,j,k]>0:
                    if not occupied:
                        occupied = True
                        front_min[i,j] = k
                    front_max[i,j] = k
    
    for i in range(dim_voxel):
        for k in range(dim_voxel):
            if top_view[i,k]>0:
                fill_flag = False
                for j in range(dim_voxel-1,-1,-1):
                    if voxel_model_256[i,j,k]>0:
                        fill_flag = True
                    else:
                        if left_min[j,k]<i and left_max[j,k]>i and front_min[i,j]<k and front_max[i,j]>k:
                            if fill_flag:
                                voxel_model_256[i,j,k]=1
                        else:
                            fill_flag = False
    
    #compress model 256 -> 64
    dim_voxel = 64
    voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    multiplier = int(256/dim_voxel)
    halfie = int(multiplier/2)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_temp[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
    
    #write voxel
    sample_voxels = np.reshape(voxel_model_temp, (dim_voxel,dim_voxel,dim_voxel,1))
    
    #sample points near surface
    batch_size = batch_size_3
    
    sample_points = np.zeros([batch_size,3],np.uint8)
    sample_values = np.zeros([batch_size,1],np.uint8)
    batch_size_counter = 0
    voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
    for j in temp_range:
        if (batch_size_counter>=batch_size): break
        for i in temp_range:
            if (batch_size_counter>=batch_size): break
            for k in temp_range:
                if (batch_size_counter>=batch_size): break
                if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
                    si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
                    sample_points[batch_size_counter,0] = si+i*multiplier
                    sample_points[batch_size_counter,1] = sj+j*multiplier
                    sample_points[batch_size_counter,2] = sk+k*multiplier
                    sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
                    voxel_model_temp_flag[i,j,k] = 1
                    batch_size_counter +=1
    if (batch_size_counter>=batch_size):
        print("64-- batch_size exceeded!")
        exceed_64_flag = 1
    else:
        exceed_64_flag = 0
        #fill other slots with random points
        while (batch_size_counter<batch_size):
            while True:
                i = random.randint(0,dim_voxel-1)
                j = random.randint(0,dim_voxel-1)
                k = random.randint(0,dim_voxel-1)
                if voxel_model_temp_flag[i,j,k] != 1: break
            si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
            sample_points[batch_size_counter,0] = si+i*multiplier
            sample_points[batch_size_counter,1] = sj+j*multiplier
            sample_points[batch_size_counter,2] = sk+k*multiplier
            sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
            voxel_model_temp_flag[i,j,k] = 1
            batch_size_counter +=1
    
    sample_points_64 = sample_points
    sample_values_64 = sample_values
    
    
    
    
    
    #compress model 256 -> 32
    dim_voxel = 32
    voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    multiplier = int(256/dim_voxel)
    halfie = int(multiplier/2)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_temp[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
    
    #sample points near surface
    batch_size = batch_size_2
    
    sample_points = np.zeros([batch_size,3],np.uint8)
    sample_values = np.zeros([batch_size,1],np.uint8)
    batch_size_counter = 0
    voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
    for j in temp_range:
        if (batch_size_counter>=batch_size): break
        for i in temp_range:
            if (batch_size_counter>=batch_size): break
            for k in temp_range:
                if (batch_size_counter>=batch_size): break
                if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
                    si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
                    sample_points[batch_size_counter,0] = si+i*multiplier
                    sample_points[batch_size_counter,1] = sj+j*multiplier
                    sample_points[batch_size_counter,2] = sk+k*multiplier
                    sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
                    voxel_model_temp_flag[i,j,k] = 1
                    batch_size_counter +=1
    if (batch_size_counter>=batch_size):
        print("32-- batch_size exceeded!")
        exceed_32_flag = 1
    else:
        exceed_32_flag = 0
        #fill other slots with random points
        while (batch_size_counter<batch_size):
            while True:
                i = random.randint(0,dim_voxel-1)
                j = random.randint(0,dim_voxel-1)
                k = random.randint(0,dim_voxel-1)
                if voxel_model_temp_flag[i,j,k] != 1: break
            si,sj,sk = sample_point_in_cube(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
            sample_points[batch_size_counter,0] = si+i*multiplier
            sample_points[batch_size_counter,1] = sj+j*multiplier
            sample_points[batch_size_counter,2] = sk+k*multiplier
            sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
            voxel_model_temp_flag[i,j,k] = 1
            batch_size_counter +=1
    
    sample_points_32 = sample_points
    sample_values_32 = sample_values
    
    #compress model 256 -> 16
    dim_voxel = 16
    voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    multiplier = int(256/dim_voxel)
    halfie = int(multiplier/2)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_temp[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
    print(voxel_model_temp)
    visualize_3d_arr(voxel_model_temp)

def visualize_3d_arr(arr):
    """ Visualize a 3 dimensional array that represents voxels
    Args:
        arr (numpy.array): a 3D array representing voxels
    """
    x = arr[0]
    y = arr[1]
    z = arr[2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(arr, edgecolor='k')
    plt.show()

def visualize_scatterplot(arr):
    x = arr[0]
    y = arr[1]
    z = arr[2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()

if __name__ == '__main__':
    # check_data()
    # voxl.voxelization('.\\square_rings\\0.obj')
    convert_to_arrays()
    # IMNET_point_sampling()
    # arr = np.array(
    #     [
    #         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #         [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    #     ]
    # )
    # visualize_3d_arr(arr)