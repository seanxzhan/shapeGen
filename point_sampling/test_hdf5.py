import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random

# The output dimension from gather_hdf5.py
dim = 16

hdf5_path = './square_rings/square_rings_vox'+str(dim)+'.hdf5'
voxel_input = h5py.File(hdf5_path, 'r')
voxel_input_voxels = voxel_input["voxels"][:]
voxel_input_points = voxel_input["points_"+str(dim)][:]
voxel_input_values = voxel_input["values_"+str(dim)][:]

if not os.path.exists("tmp"):
	os.makedirs("tmp")

# the original model dimension
res = 64

for idx in range(10):
	vox = voxel_input_voxels[idx,:,:,:,0]*255
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_vox_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_vox_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_vox_3.png",img3)

	vox = np.zeros([res,res,res],np.uint8)
	batch_points_int = voxel_input_points[idx]
	batch_values = voxel_input_values[idx]
	vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	print(vox.shape)
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_p"+str(dim)+"_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_p"+str(dim)+"_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_p"+str(dim)+"_3.png",img3)
	