# -*- coding: utf-8 -*-
"""
Functions that can voxelization the *.ply files
python 2.7
Author: Chaoqun Jiang
Home Page: http://mcoder.cc
"""

# import pyassimp
import pywavefront as pf
import numpy as np
import operator
import pyflann
import json
import time
import os

def voxelization(
    filename,
    outputJsonPath = '.\\voxel_json\\',
    outputNumpyPath = '.\\voxel_numpy\\',
    coef = 1.0,
    size = (16,16,16)):
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
        coef: used to judge if the point is 1 or 0
        size: a tuple with 3 integer, default is (192, 192, 200)
    Return:
        None: if no voxel has calculated, return None
        numpy.ndarray:  if the voxel has been calculated, return ndarray
    """
    if len(size) != 3:
        print("The argument \" size \" should has three integer")
        return

    # scene = pyassimp.load(filename)     # import scene
    scene = pf.Wavefront(filename, collect_faces=True)
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

    scene_vertices = np.asarray(scene.vertices)
    for index in range(meshes_count):
        _meshVoxel(start, edge, scene_vertices, voxel, coef, str(index))
    print("calculate all meshes voxel finished!")

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

    scene_vertices = np.asarray(scene.vertices)
    xmax, ymax, zmax = np.amax( scene_vertices, axis = 0 )
    xmin, ymin, zmin = np.amin( scene_vertices, axis = 0 )

    # print("Bounding box: ",xmax, ymax, zmax, xmin, ymin, zmin)
    print("Bounding box: xmax: {0}, ymax: {1}, zmax:{2}, xmin: {3}, ymin: {4}, zmin: {5}".format(
        xmax, ymax, zmax, xmin, ymin, zmin))
    return (xmax, ymax, zmax, xmin, ymin, zmin)

def _meshVoxel(startpoint, edge, scene_vertices, voxel, coef = 1.0, str = "0"):
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
    vertices = scene_vertices    #  np.array n x 3

    print(vertices)

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
                    startpoint[2] + z * edge]], dtype = np.float64)
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

if __name__ == '__main__':
    vox_arr = voxelization('.\\square_rings\\0.obj')
    print(vox_arr)