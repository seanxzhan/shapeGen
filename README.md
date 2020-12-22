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

### Questions

- How to write some code that generates a 3D model? 

Write a python script that generates arrays that represent 3D shapes. 

- What format should those 3D models be in? 

.mat

- What dependencies do I need?

```scipy.io```, ```matplotlib```

I will be using https://github.com/czq142857/IM-NET for training, and the input data should be in hdf5 format, so I probably need ```h5py``` at some point to convert what I generate (arrays??) into hdf5 format?? 

--- 

### Dev

#### Create virtual env

The project is developed in Windows 10

```
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```
