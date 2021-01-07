import numpy as np
import cv2
import os
import h5py
import random

is_train_set = False
voxel_input_dir = './square_rings'

folder_name = "square_rings_dataset"
if not os.path.exists(folder_name):
	os.makedirs(folder_name)
class_name = "square_rings"

if is_train_set:
	num_shapes = 800
else:
	num_shapes = 200

# write voxels
vox_size = 16 # remember we sampled at 16x16x16 resolution in gather_hdf5.py
# batch_size_3 = 8*8*8*4
batch_size_3 = 16**3

# output hdf5
if is_train_set:
	hdf5_file = h5py.File(folder_name+'/'+class_name+"_vox64_train.hdf5", 'w')
else:
	hdf5_file = h5py.File(folder_name+'/'+class_name+"_vox64_test.hdf5", 'w')
# hdf5_file.create_dataset("pixels", [num_shapes,num_view,view_size,view_size], np.uint8, compression=9)
hdf5_file.create_dataset("voxels", [num_shapes,vox_size,vox_size,vox_size,1], np.uint8, compression=9)
hdf5_file.create_dataset("points_"+str(vox_size), [num_shapes,batch_size_3,3], np.uint8, compression=9)
hdf5_file.create_dataset("values_"+str(vox_size), [num_shapes,batch_size_3,1], np.uint8, compression=9)

counter = 0

input_len = 1000
if is_train_set:
	start_len = 0
	target_len = int(input_len*0.8)
else:
	start_len = int(input_len*0.8)
	target_len = input_len-start_len

# input hdf5
voxel_hdf5_dir1 = voxel_input_dir+'/'+class_name+'_vox'+str(vox_size)+'.hdf5'
voxel_hdf5_file1 = h5py.File(voxel_hdf5_dir1, 'r')
voxel_hdf5_voxels = voxel_hdf5_file1['voxels'][:]
voxel_input_points = voxel_hdf5_file1['points_'+str(vox_size)][:]
voxel_input_values = voxel_hdf5_file1['values_'+str(vox_size)][:]
voxel_hdf5_file1.close()

print(counter,num_shapes)

hdf5_file["voxels"][counter:counter+target_len] = voxel_hdf5_voxels[start_len:start_len+target_len]
hdf5_file["points_"+str(vox_size)][counter:counter+target_len] = voxel_input_points[start_len:start_len+target_len]
hdf5_file["values_"+str(vox_size)][counter:counter+target_len] = voxel_input_values[start_len:start_len+target_len]

counter += target_len

assert(counter==num_shapes)
hdf5_file.close()


