## ShapeGen

This project has two parts:

- A procedural model that generates a dataset of one type of procedural 3D shapes. 

- A generative model that trains on the dataset and generates those shapes. 

---

### Terms

- **Procedural Modeling** is the process of creating 3D models from a set of rules. 
What 3D format? 
https://www.e-education.psu.edu/geogvr/node/534
https://link.springer.com/article/10.1007/s00371-018-1589-4?shared-article-renderer

---

### Roadmap

1. Implement https://github.com/czq142857/IM-NET (get data, understand and run the model)

    - Data preparation takes in .mat files. Need to figure out this workflow to setup my own dataset. 

2. Use Blender + pyton to create a dataset of simple shapes (rectangular rings). 

    - These rectangular rings will have dimension (random.uniform(0.1, 0.4), 1, 1). The **y** and **z** dimensions are the same because it'd be interesting to see if the generative network can learn this property. The rings will also be kind of flat, so the output of the network should also be flat. Note that the size of the hole varies. 

    - Outputs .obj & .mtl(optional)

3. Voxelize these objects (.obj) into 3D arrays, then encode the dataset as .hdf5

4. Run this dataset of shapes through the model

5. If successful, go back to Blender and make some interesting procedural shapes.

---

### Questions

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



---

### Files

**generate_data.py**: generate a dataset of procedural shapes using bpy

**visualize.py**: visualize a .mat file

---

### Dev

#### Create virtual env

The project is developed in Windows 10

```
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

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



