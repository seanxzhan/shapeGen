# -*- coding: utf-8 -*-
"""
Functions that can voxelization the *.ply files
python 2.7
Author: Chaoqun Jiang
Home Page: http://mcoder.cc
"""

import pyassimp
import numpy as np
import operator
import pyflann
import json
import binvox_rw
import time
import os

def voxelization(
    filename,
    outputJsonPath = '../voxel_json/',
    outputNumpyPath = '../voxel_numpy/',
    outputBinvoxPath = '../voxel_binvox/',
    coef = 1.0,
    size = (192,192,200)):
    """ function voxelization
        This function load a *.ply model file, and convert it into a voxel.
        And export in two formats.
        numpy formats: just use numpy import, a array has shape (192, 192, 200)
        json format: a numpy format reshape to (-1,) and attribute name is 'array'
    Args:
        filename:   a relative file path to the *.ply file
        outputJsonPath: a relative floder path to save voxel in json format
        outputNumpyPath: a relative floder path to save voxel in numpy format
            Note: The directory should already be created. Or it will throw IOError
        outputBinvoxPath: a relative floder path to save in binvox format
        coef: used to judge if the point is 1 or 0
        size: a tuple with 3 integer, default is (192, 192, 200)
    Return:
        None: if no voxel has calculated, return None
        numpy.ndarray:  if the voxel has been calculated, return ndarray
    """
    if len(size) != 3:
        print("The argument \" size \" should has three integer")
        return

    scene = pyassimp.load(filename)     # import scene
    meshes_count = len(scene.meshes)    # the count of meshes
    if meshes_count < 1:
        print("Error! The model file has no meshes in it")
        return

    voxel_width = size[0]
    voxel_height = size[1]
    voxel_length = size[2]

    voxel = np.zeros( shape = (voxel_width, voxel_height, voxel_length),
        dtype = np.int8)         # Creat a zeros ndarray
    print("Program is manipulating model: ", filename)
    print("Program will create voxel in shape", size)

    boundingbox = _getBoundingBox(scene)    # get the bounding box of scene

    # calculate each voxel's edge length
    center = np.array( [ (boundingbox[0] + boundingbox[3]) / 2,
                        (boundingbox[1] + boundingbox[4]) / 2,
                        (boundingbox[2] + boundingbox[5]) /2] )

    x_edge = (boundingbox[0] - boundingbox[3]) / voxel_width
    y_edge = (boundingbox[1] - boundingbox[4]) / voxel_height
    z_edge = (boundingbox[2] - boundingbox[5]) / voxel_length
    edge = max(x_edge, y_edge, z_edge)      # use the max as edge
    print ("x_edge: {0}, y_edge: {1}, z_edge: {2}, edge: {3}".format(
        x_edge, y_edge, z_edge, edge))

    # set the (voxel_width // 2, voxel_height // 2, voxel_length // 2)'s
    # position is center. So we can get other voxel box's voxel box.
    # At here, we calculate the start voxel box's center position.
    start = center - np.array([voxel_width // 2 * edge,
        voxel_height // 2 * edge, voxel_length // 2 * edge])

    #print("center", center, "start", start)
    print("center: {0}, staet: {1}".format(center, start))

    for index in range(meshes_count):
        _meshVoxel(start, edge, scene.meshes[index], voxel, coef, str(index))
    print("calculate all meshes voxel finished!")

    # save voxel files
    _saveVoxel(filename,
        outputJsonPath, outputNumpyPath, outputBinvoxPath, voxel)
    return voxel

def _getBoundingBox(scene):
    """give a assimp scene, get it bounding box
            It will bounding all meshes in the mesh.
        Args:
            scene: assimp scene
        Returns:
            bounding box ( xmax, ymax, zmax, xmin, ymin, zmin )
            6 num represent 6 faces.
    """
    if len(scene.meshes) == 0:
        print("scene's meshes attribute has no mesh")
        return (0,0,0,0,0,0)

    mesh_1 = scene.meshes[0]
    xmax, ymax, zmax = np.amax( mesh_1.vertices, axis = 0 )
    xmin, ymin, zmin = np.amin( mesh_1.vertices, axis = 0 )

    for index in range(1,len(scene.meshes)):
        mesh_t = scene.meshes[index]
        xmax_t, ymax_t, zmax_t = np.amax( mesh_t.vertices, axis = 0)
        xmin_t, ymin_t, zmin_t = np.amin( mesh_t.vertices, axis = 0)

        if xmax_t > xmax:   xmax = xmax_t
        if ymax_t > ymax:   ymax = ymax_t
        if zmax_t > zmax:   zmax = zmax_t
        if xmin_t < xmin:   xmin = xmin_t
        if ymin_t < ymin:   ymin = ymin_t
        if zmin_t < zmin:   zmin = zmin_t

    # print("Bounding box: ",xmax, ymax, zmax, xmin, ymin, zmin)
    print("Bounding box: xmax: {0}, ymax: {1}, zmax:{2}, xmin: {3}, ymin: {4}, zmin: {5}".format(
        xmax, ymax, zmax, xmin, ymin, zmin))
    return (xmax, ymax, zmax, xmin, ymin, zmin)

def _meshVoxel(startpoint, edge, mesh, voxel, coef = 1.0, str = "0"):
    """ mesh voxel function
    change numpy.ndarray's 0 to 1 acounding to mesh and scene'bounding box
    Args:
        startpoint: numpy.ndarray with shape of (3,)
        edge: the voxel box's edge length
        mesh: pyassimp mesh
        voxel: numpy.ndarray
        coef: used to judge if this point is 1
        str: the string you want to split each mesh
    """
    vertices = mesh.vertices    #  np.array n x 3

    #print("The mesh ", str," has vertices: ", vertices.shape)
    print("The mesh {0} has vertices {1}".format(str, vertices.shape))

    # KDtree
    flann = pyflann.FLANN()     # create a FLANN object
    params = flann.build_index(vertices, algorithm = "kdtree", trees = 4)

    # iterate to calculate the voxel value
    # if there is a point close to the center, there is 1, otherwise, no changes
    width, height, length = voxel.shape
    start_time = time.time()
    landmark = coef * edge

    for x in range(width):
        for y in range(height):
            for z in range(length):
                # for each voxel center
                voxel_center = np.array([[
                    startpoint[0] + x * edge,
                    startpoint[1] + y * edge,
                    startpoint[2] + z * edge]],dtype = np.float32)
                result, dists = flann.nn_index(voxel_center, 1,
                    checks = params["checks"])
                index = result[0]
                vertex = vertices[index,:]  # get nearest neighbor
                distance = np.sqrt(((vertex - voxel_center) ** 2).sum())

                if distance <= landmark:
                    voxel[x,y,z] = 1

#    print("The mesh" + str +" process successfully in " ,
#        (time.time() - start_time), " s")
    print("The mesh {0} process successfully in {1}s".format(
        str, (time.time() - start_time)))

def _saveVoxel(filename,
    outputJsonPath, outputNumpyPath, outputBinvoxPath, voxel):
    """ save voxel
        Save the voxel into file.
    Args:
        filename:   the filename of import.
        outputJsonPath: path to save json.
        outputNumpyPath: path to save numpy.
        outputBinvoxPath: path to save binvox.
        voxel: numpy.ndarray
    """
    startPoint = 0
    if filename.rfind("/") != -1:
        startPoint = filename.rfind("/") + 1

    filename = filename[startPoint:filename.rfind('.')]  # cut the format end
    # save npy
    #voxel.tofile(outputNumpyPath + filename + ".numpy")
    np.save(os.path.join( outputNumpyPath, filename ) + ".npy", voxel)

    # save binvox
    bool_voxel = voxel.astype(np.bool)
    binvox = binvox_rw.Voxels(
        data = bool_voxel,
        dims = list(voxel.shape),
        translate = [0.0, 0.0, 0.0],
        scale = 1.0,
        axis_order = 'xzy')
    fp = open(os.path.join( outputBinvoxPath, filename ) + ".binvox", 'wb+')
    fp.truncate()
    binvox.write(fp)
    fp.close()

    # save json
    array = voxel.reshape(-1,)
    json_str = json.dumps(array.tolist())
    json_file = open(os.path.join( outputJsonPath, filename ) + ".json", "w+")
    json_file.truncate()            # 清空当前文件的内容
    json_file.write(json_str)
    json_file.close()