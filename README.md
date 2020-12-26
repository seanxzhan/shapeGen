## ShapeGen

This project has two parts:

- A procedural model that generates a dataset of one type of procedural 3D shapes. 

- A generative model that trains on the dataset and generates those shapes. 

---

### Terms

- **Procedural Modeling** is the process of creating 3D models from a set of rules. 
What 3D format? 
https://www.e-education.psu.edu/geogvr/node/534

---

### Roadmap

1. Implement https://github.com/czq142857/IM-NET (get data, understand and run the model)

    - Data preparation takes in .mat files. 

2. Use Blender + pyton to create a dataset of cuboids. 

    - These cuboids will have dimension (random.uniform(0.1, 0.4), 1, 1). The **y** and **z** dimensions are the same because it'd be interesting to see if the generative network can learn this property. The cuboids will also be kind of flat, so the output of the network should also be flat.

    - Outputs .obj & .mtl(optional)

3. Somehow convert .obj files to .mat files

4. Run my own dataset of shapes through the model

5. If successful, go back to Blender and make more interesting shapes.

---

### Questions

- How to write some code that generates a 3D model? 

Write a python script that generates arrays that represent 3D shapes. 

- What format should those 3D models be in? 

First Blender will probably output .obj, then conver that to .mat

- What dependencies do I need?

```scipy.io```, ```matplotlib```

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



