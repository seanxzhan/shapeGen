import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random

# class_name = "03001627_chair"
# hdf5_path = class_name+"/"+class_name[:8]+"_vox256.hdf5"
hdf5_path = './square_rings/square_rings_vox16.hdf5'
voxel_input = h5py.File(hdf5_path, 'r')
voxel_input_voxels = voxel_input["voxels"][:]
# voxel_input_points_2 = voxel_input["points_2"][:]
# voxel_input_values_2 = voxel_input["values_2"][:]
# voxel_input_points_4 = voxel_input["points_4"][:]
# voxel_input_values_4 = voxel_input["values_4"][:]
# voxel_input_points_8 = voxel_input["points_8"][:]
# voxel_input_values_8 = voxel_input["values_8"][:]
voxel_input_points_16 = voxel_input["points_16"][:]
voxel_input_values_16 = voxel_input["values_16"][:]

if not os.path.exists("tmp"):
	os.makedirs("tmp")

res = 64

for idx in range(10):
	vox = voxel_input_voxels[idx,:,:,:,0]*255
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_vox_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_vox_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_vox_3.png",img3)

	# vox = np.zeros([res,res,res],np.uint8)
	# batch_points_int = voxel_input_points_2[idx]
	# batch_values = voxel_input_values_2[idx]
	# vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	# img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	# img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	# img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	# cv2.imwrite("tmp/"+str(idx)+"_p2_1.png",img1)
	# cv2.imwrite("tmp/"+str(idx)+"_p2_2.png",img2)
	# cv2.imwrite("tmp/"+str(idx)+"_p2_3.png",img3)
	
	# vox = np.zeros([res,res,res],np.uint8)
	# batch_points_int = voxel_input_points_4[idx]
	# batch_values = voxel_input_values_4[idx]
	# vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	# img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	# img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	# img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	# cv2.imwrite("tmp/"+str(idx)+"_p4_1.png",img1)
	# cv2.imwrite("tmp/"+str(idx)+"_p4_2.png",img2)
	# cv2.imwrite("tmp/"+str(idx)+"_p4_3.png",img3)
	
	# vox = np.zeros([res,res,res],np.uint8)
	# batch_points_int = voxel_input_points_8[idx]
	# batch_values = voxel_input_values_8[idx]
	# vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	# img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	# img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	# img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	# cv2.imwrite("tmp/"+str(idx)+"_p8_1.png",img1)
	# cv2.imwrite("tmp/"+str(idx)+"_p8_2.png",img2)
	# cv2.imwrite("tmp/"+str(idx)+"_p8_3.png",img3)

	vox = np.zeros([res,res,res],np.uint8)
	batch_points_int = voxel_input_points_16[idx]
	batch_values = voxel_input_values_16[idx]
	vox[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
	img1 = np.clip(np.amax(vox, axis=0)*256, 0,255).astype(np.uint8)
	img2 = np.clip(np.amax(vox, axis=1)*256, 0,255).astype(np.uint8)
	img3 = np.clip(np.amax(vox, axis=2)*256, 0,255).astype(np.uint8)
	cv2.imwrite("tmp/"+str(idx)+"_p16_1.png",img1)
	cv2.imwrite("tmp/"+str(idx)+"_p16_2.png",img2)
	cv2.imwrite("tmp/"+str(idx)+"_p16_3.png",img3)
	