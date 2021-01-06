## ShapeGen

This project generates a type of procedural 3D shapes (currently rectangular rings) and trains on the dataset and generates those shapes. 

---

### Roadmap

1. Run https://github.com/czq142857/IM-NET (get data, understand and run the model)

    - Data preparation takes in .mat files. Need to figure out this workflow to setup my own dataset. EDIT: workflow is described below. 

2. Use Blender + python to create a dataset of simple shapes (rectangular rings). 

    - These rectangular rings will have dimension (random.uniform(0.1, 0.4), 1, 1). The **y** and **z** dimensions are the same because it'd be interesting to see if the generative network can learn this property. The rings will also be kind of flat, so the output of the network should also be flat. Note that the size of the hole varies. 

    - Outputs .obj & .mtl(optional)

3. Voxelize these objects (.obj) into 3D arrays, then encode the dataset as .hdf5

    - The function ```voxelize_and_save()``` in **preprocess.py** will save these wavefront objects to a bunch of **.npy** arrays. I will then modify pythons scripts IM-NET/point_sampling to load and encode these arrays into .hdf5 following IM-NET's point sampling method.

4. Run this dataset of shapes through the model

    - **Current Problem**: The autoencoder doesn't seem to be learning when training grid size (```real_size```) is bigger than 4. It "gets stuck". **Possible reason**: there are too few data points in my dataset, and the generated .ply ends up being a cube **Possible solutions**: 1. treat my dataset as if it's a blurry dog image ==> add a transposed convolution layer to "lift" the information to a higher dimension. Not exactly sure if this will backfire and the model will get even worse 2. save my wavefront objects as 64^3 voxels instead of 16^3 voxels.

5. If successful, go back to Blender and make some interesting procedural shapes.

---

### Questions

**What's procedural modeling?**

**Procedural Modeling** is the process of creating 3D models from a set of rules. 
https://www.e-education.psu.edu/geogvr/node/534
https://link.springer.com/article/10.1007/s00371-018-1589-4?shared-article-renderer

**How to write some code that generates a 3D model?**

~~Write a python script that generates arrays that represent 3D shapes.~~ Write a python script in Blender then export mesh to wavefront obj. Here's a paper related to procedural model generation: https://link.springer.com/article/10.1007/s00371-018-1589-4?shared-article-renderer

**What are the 'b' and 'bi' fields in the dataset mentioned in https://github.com/czq142857/IM-NET?**

After visualizing data in 'bi' and voxel_model_256 as voxels in matplotlib, it seems like 'bi' is a very rough base for a model (very pixelated), and 'b' contains some sort of encoding that sculpts the base model and makes it more detailed as a result of point sampling. This is illustrated in preprocess.py -- IMNET_point_sampling.

**To what format should I convert the .obj files?**

Reading code in ```point_sampling.py``` from IMNET tells me that I need to convert these wavefront objects to 3 dimensional arrays and encode these objects to a hdf5 (hierarchical data format). Converting to 3D arrays makes sense because (3D shapes -> 3D arrays of voxels) == (2D images -> 2D arrays of pixels). Boom.  

**What are the fields in wavefront obj? How to use these fields to turn obj into a 3D array?**

- ```v```: vertices
- ```vn```: vertex normal
- ```f```: polygon face element

If we want a (16, 16, 16) array, we can construct a 16 by 16 by 16 coordinate system. The 3D can be initialized as ```coors = numpy.zeros((16, 16, 16))```. There are 16x16x16 = 4096 cubes of size 1 by 1 by 1. If the center of a cube (say a cube at (1, 2, 3)) lies inside of our shape, the coordinate ```coors[1][2][3]``` will be set to ```1```. 

Ahah the above is actually very similar to the marching cubes algorithm but reversed and crude lol. 

In this project, I will use a package ```mesh-to-sdf```. It takes in a wavefront object and outputs an ndarray of SDF (signed distance field) values. If the value at a position P is positive, then point P lies outside of the object, and if the value is negative, then P is inside the object. With this knowledge, we can simply convert the sdf output to a 3d array with 1s and 0s to represent voxels. (Spoiler alert, the network I will be using (IM-NET) actually learns whether a given coordinate in a grid is inside / outside of the 3D object)

**How does the dimension of voxels array impact result?**

I'm currently using (16, 16, 16) arrays to represent my square rings, but I'm worried that this is too small. I used IM-NET's progressive point sampling, and it brought it down to (8, 8, 8), (4, 4, 4), and (2, 2, 2), so the network will [gradually] learn from 2^3 to 4^3 to 8^3. We will see how this goes. 

If the network doesn't learn, I might need to convert my square rings wavefront objects to (64, 64, 64), but the conversion might take a long time. 

EDIT: The above concern is related to the current problem in Roadmap step 4. 

---

### Files

**generate_data.py**: generate a dataset of procedural shapes (rectangular rings) using ```bpy``` in Blender

**preprocess.py**: voxelize the generated wavefront objects and save to .npy files

**./point_sampling**: create .hdf5 datasets using the .npy files

**./IMGAN**: autoencoder and generator network from IM-NET

**./IMSVR**: single view reconstruction network from IM-NET

---

### Dev log

Followed README of https://github.com/czq142857/IM-NET. The data that this repo references seems to be processed data in .mat. No clue what the 'b' and 'bi' fields in the .mat file mean. 

Went to ShapeNet website, downloaded their data (massive zip), found out that these 3D models should be in .obj format, there's an .mtl file that goes along with it. Heard of Blender, guessing that maybe I can write some script in Blender to generate these files.

Found some cool reddit people doing procedural modeling: 

- https://www.reddit.com/r/proceduralgeneration/comments/4mn9gj/monthly_challenge_7_june_2016_procedural/

- https://github.com/a1studmuffin/SpaceshipGenerator

- https://github.com/aaronjolson/Blender-Python-Procedural-Level-Generation

Blender python documentation:

https://docs.blender.org/api/current/index.html

Some tutorials on Blender + python: 

https://www.mertl-research.at/ceonwiki/doku.php?id=software:kicad:3d_package_with_blender

**Extruding surfaces**:

- https://blenderartists.org/t/extruding-already-selected-faces-at-different-random-amounts-using-python-scripting/1163223/2

- https://blender.stackexchange.com/questions/115397/extrude-in-python

IM-NET readme: Training on the 13 ShapeNet categories takes about 4 days on one GeForce RTX 2080 Ti GPU. ==> I decided to voxelize the wavefront objects into 16 by 16 by 16 voxel representations to reduce training time (I want to get the entire workflow worked out first before working w/ more complex shapes and higher resolutions). 

Simplified point_sampling to fit my one-model-type dataset. 

The autoencoder doesn't seem to be learning when training grid size (```real_size```) is bigger than 4. **Possible reason**: there are too few data points in my dataset. **Possible solutions**: 1. treat my dataset as if it's a blurry dog image ==> add a transposed convolution layer to "lift" the information to a larger resolution. 2. save my wavefront objects as 64^3 voxels instead of 16^3 voxels.


'''
cd IMGAN

python main.py --ae --train --epoch 20 --real_size 2 --batch_size_input 8

python main.py --ae --train --epoch 40 --real_size 4 --batch_size_input 64

python main.py --ae --train --epoch 80 --real_size 8 --batch_size_input 256

python main.py --ae --train --epoch 50 --real_size 16 --batch_size_input 2048

python main.py --ae

python main.py --train --epoch 10000

python main.py
'''