import numpy as np
# import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json
from multiprocessing import Process, Queue
import queue
import time
#import mcubes
import matplotlib.pyplot as plt
import re

dim = 16

# batch_size_1 = 16*16*16
# batch_size_2 = 16*16*16
# batch_size_3 = 16*16*16*4
# batch_size_1 = 2*2*2 
# batch_size_2 = 4*4*4
# batch_size_3 = 4*4*4*4
batch_size_3 = 8*8*8*4
# batch_size_3 = 4*4*4

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

'''
#do not use progressive sampling (center2x2x2 -> 4x4x4 -> 6x6x6 ->...)
#if sample non-center points only for inner(1)-voxels,
#the reconstructed model will have railing patterns.
#since all zero-points are centered at cells,
#the model will expand one-points to a one-planes.
'''
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

def get_points_from_vox(q, path):
	count = 0
	for filename in os.listdir(path):
		count += 1

		try:
			# voxel_model_16_original = np.load(path + '/' + filename)
			# voxel_model_16_original = voxel_model_16_original.astype('uint8')
			voxel_model_64 = np.load(path + '/' + filename)
			# voxel_model_64 = voxel_model_64.astype('uint8')
		except:
			print("error in loading")
			exit(-1)
		
		# #carve the voxels from side views:
		# #top direction = Y(j) positive direction
		# dim_voxel = 16
		# top_view = np.max(voxel_model_16_original, axis=1)
		# left_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
		# left_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
		# front_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
		# front_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
		
		# for j in range(dim_voxel):
		# 	for k in range(dim_voxel):
		# 		occupied = False
		# 		for i in range(dim_voxel):
		# 			if voxel_model_16_original[i,j,k]>0:
		# 				if not occupied:
		# 					occupied = True
		# 					left_min[j,k] = i
		# 				left_max[j,k] = i
		
		# for i in range(dim_voxel):
		# 	for j in range(dim_voxel):
		# 		occupied = False
		# 		for k in range(dim_voxel):
		# 			if voxel_model_16_original[i,j,k]>0:
		# 				if not occupied:
		# 					occupied = True
		# 					front_min[i,j] = k
		# 				front_max[i,j] = k
		
		# for i in range(dim_voxel):
		# 	for k in range(dim_voxel):
		# 		if top_view[i,k]>0:
		# 			fill_flag = False
		# 			for j in range(dim_voxel-1,-1,-1):
		# 				if voxel_model_16_original[i,j,k]>0:
		# 					fill_flag = True
		# 				else:
		# 					if left_min[j,k]<i and left_max[j,k]>i and front_min[i,j]<k and front_max[i,j]>k:
		# 						if fill_flag:
		# 							voxel_model_16_original[i,j,k]=1
		# 					else:
		# 						fill_flag = False		

		#compress model 16 -> 8
		#EDIT: compress model 64 -> 16
		dim_voxel = 16
		voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		# multiplier = int(16/dim_voxel)
		multiplier = int(64/dim_voxel)
		halfie = int(multiplier/2)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_temp[i,j,k] = np.max(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		#write voxel
		# visualize_3d_arr(voxel_model_temp)
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
						si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
						sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
						voxel_model_temp_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			# print("8-- batch_size exceeded!")
			# exceed_8_flag = 1
			print("16-- batch_size exceeded!")
			exceed_16_flag = 1
		else:
			# exceed_8_flag = 0
			exceed_16_flag = 0
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_temp_flag[i,j,k] != 1: break
				si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		# sample_points_8 = sample_points
		# sample_values_8 = sample_values
		sample_points_16 = sample_points
		sample_values_16 = sample_values

		# print(sample_points_8)
		# print(sample_values_8)
		
		# #compress model 16 -> 4
		# dim_voxel = 4
		# voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		# multiplier = int(16/dim_voxel)
		# halfie = int(multiplier/2)
		# for i in range(dim_voxel):
		# 	for j in range(dim_voxel):
		# 		for k in range(dim_voxel):
		# 			voxel_model_temp[i,j,k] = np.max(voxel_model_16_original[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		# #sample points near surface
		# batch_size = batch_size_2
		
		# sample_points = np.zeros([batch_size,3],np.uint8)
		# sample_values = np.zeros([batch_size,1],np.uint8)
		# batch_size_counter = 0
		# voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		# temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		# for j in temp_range:
		# 	if (batch_size_counter>=batch_size): break
		# 	for i in temp_range:
		# 		if (batch_size_counter>=batch_size): break
		# 		for k in temp_range:
		# 			if (batch_size_counter>=batch_size): break
		# 			if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
		# 				si,sj,sk = sample_point_in_cube(voxel_model_16_original[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
		# 				sample_points[batch_size_counter,0] = si+i*multiplier
		# 				sample_points[batch_size_counter,1] = sj+j*multiplier
		# 				sample_points[batch_size_counter,2] = sk+k*multiplier
		# 				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
		# 				voxel_model_temp_flag[i,j,k] = 1
		# 				batch_size_counter +=1
		# if (batch_size_counter>=batch_size):
		# 	print("4-- batch_size exceeded!")
		# 	exceed_4_flag = 1
		# else:
		# 	exceed_4_flag = 0
		# 	#fill other slots with random points
		# 	while (batch_size_counter<batch_size):
		# 		while True:
		# 			i = random.randint(0,dim_voxel-1)
		# 			j = random.randint(0,dim_voxel-1)
		# 			k = random.randint(0,dim_voxel-1)
		# 			if voxel_model_temp_flag[i,j,k] != 1: break
		# 		si,sj,sk = sample_point_in_cube(voxel_model_16_original[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
		# 		sample_points[batch_size_counter,0] = si+i*multiplier
		# 		sample_points[batch_size_counter,1] = sj+j*multiplier
		# 		sample_points[batch_size_counter,2] = sk+k*multiplier
		# 		sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
		# 		voxel_model_temp_flag[i,j,k] = 1
		# 		batch_size_counter +=1
		
		# sample_points_4 = sample_points
		# sample_values_4 = sample_values
				
		# #compress model 16 -> 2
		# dim_voxel = 2
		# voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		# multiplier = int(16/dim_voxel)
		# halfie = int(multiplier/2)
		# for i in range(dim_voxel):
		# 	for j in range(dim_voxel):
		# 		for k in range(dim_voxel):
		# 			voxel_model_temp[i,j,k] = np.max(voxel_model_16_original[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		# #sample points near surface
		# batch_size = batch_size_1
		
		# sample_points = np.zeros([batch_size,3],np.uint8)
		# sample_values = np.zeros([batch_size,1],np.uint8)
		# batch_size_counter = 0
		# for i in range(dim_voxel):
		# 	for j in range(dim_voxel):
		# 		for k in range(dim_voxel):
		# 			si,sj,sk = sample_point_in_cube(voxel_model_16_original[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
		# 			sample_points[batch_size_counter,0] = si+i*multiplier
		# 			sample_points[batch_size_counter,1] = sj+j*multiplier
		# 			sample_points[batch_size_counter,2] = sk+k*multiplier
		# 			sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
		# 			batch_size_counter +=1
		# if (batch_size_counter!=batch_size):
		# 	print("batch_size_counter!=batch_size")
		
		# sample_points_2 = sample_points
		# sample_values_2 = sample_values
		
		name = re.search('(.*).npy', filename).group(1)
		name = int(name)
		# q.put([name,exceed_8_flag,exceed_4_flag,sample_points_8,sample_values_8,sample_points_4,sample_values_4,sample_points_2,sample_values_2,sample_voxels])
		# q.put([name,exceed_8_flag,sample_points_8,sample_values_8,sample_voxels])
		print("put")
		q.put([name,exceed_16_flag,sample_points_16,sample_values_16,sample_voxels])

def list_image(root, exts):
	image_list = []
	cat = {}
	for path, subdirs, files in os.walk(root):
		for fname in files:
			fpath = os.path.join(path, fname)
			suffix = os.path.splitext(fname)[1].lower()
			if os.path.isfile(fpath) and (suffix in exts):
				if path not in cat:
					cat[path] = len(cat)
				image_list.append((os.path.relpath(fpath, root), cat[path]))
	return image_list

if __name__ == '__main__':
	obj_name = 'square_rings'

	if not os.path.exists(obj_name):
		os.makedirs(obj_name)

	#name of output file
	# hdf5_path = obj_name+'/'+obj_name+'_vox16.hdf5'
	hdf5_path = obj_name+'/'+obj_name+'_vox16.hdf5'

	#obj_list
	# fout = open(obj_name+'/'+obj_name+'_vox16.txt','w',newline='')
	
	#record statistics
	fstatistics = open(obj_name+'/statistics.txt','w',newline='')
	exceed_4 = 0 #32
	exceed_8 = 0 #64
	exceed_16 = 0
	
	#map processes
	q = Queue()
	# path = '../square_rings_vox'
	path = '../square_rings_vox_64'
	# workers = [Process(target=get_points_from_vox, args = (q, list_of_names)) for list_of_names in list_of_list_of_names]
	workers = [Process(target=get_points_from_vox, args = (q, path))]

	for p in workers:
		p.start()

	num_items = 1000

	#reduce process
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("voxels", [num_items,dim,dim,dim,1], np.uint8)
	# hdf5_file.create_dataset("points_2", [num_items,batch_size_1,3], np.uint8)
	# hdf5_file.create_dataset("values_2", [num_items,batch_size_1,1], np.uint8)
	# hdf5_file.create_dataset("points_4", [num_items,batch_size_2,3], np.uint8)
	# hdf5_file.create_dataset("values_4", [num_items,batch_size_2,1], np.uint8)
	hdf5_file.create_dataset("points_16", [num_items,batch_size_3,3], np.uint8)
	hdf5_file.create_dataset("values_16", [num_items,batch_size_3,1], np.uint8)



	while True:
		item_flag = True
		try:
			# idx,exceed_8_flag,exceed_4_flag,sample_points_8,sample_values_8,sample_points_4,sample_values_4,sample_points_2,sample_values_2,sample_voxels = q.get(True, 1.0)
			idx,exceed_16_flag,sample_points_16,sample_values_16,sample_voxels = q.get(True, 1.0)
		except queue.Empty:
			item_flag = False
		
		if item_flag:
			#process result
			# exceed_4+=exceed_4_flag
			# exceed_8+=exceed_8_flag
			exceed_16+=exceed_16_flag
			hdf5_file["points_16"][idx,:,:] = sample_points_16
			hdf5_file["values_16"][idx,:,:] = sample_values_16
			# hdf5_file["points_4"][idx,:,:] = sample_points_4
			# hdf5_file["values_4"][idx,:,:] = sample_values_4
			# hdf5_file["points_2"][idx,:,:] = sample_points_2
			# hdf5_file["values_2"][idx,:,:] = sample_values_2
			hdf5_file["voxels"][idx,:,:,:,:] = sample_voxels
		
		allExited = True
		for p in workers:
			if p.exitcode is None:
				allExited = False
				break
		if allExited and q.empty():
			break

	fstatistics.write("total: "+str(num_items)+"\n")
	# fstatistics.write("exceed_4: "+str(exceed_4)+"\n")
	# fstatistics.write("exceed_4_ratio: "+str(float(exceed_4)/num_items)+"\n")
	# fstatistics.write("exceed_8: "+str(exceed_8)+"\n")
	# fstatistics.write("exceed_8_ratio: "+str(float(exceed_8)/num_items)+"\n")
	fstatistics.write("exceed_16: "+str(exceed_16)+"\n")
	fstatistics.write("exceed_16_ratio: "+str(float(exceed_16)/num_items)+"\n")
	
	fstatistics.close()
	hdf5_file.close()
	print("finished")


