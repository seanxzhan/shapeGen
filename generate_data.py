import bpy # import this to enable python
import bmesh
import random
import os
from math import radians

# use alt-p to run script, output will be in Window -> Toggle System Console

scene = bpy.context.scene

# delete all objects remaining in the scene since last render
def delete_all():
    # select all then delete
    for obj in bpy.data.objects:
        if obj.name.startswith("RING") or obj.name.startswith("Cube"):
            mesh = bpy.data.meshes[obj.name]
            bpy.data.meshes.remove(mesh)

def create_cam_body():
    cam_body = bpy.data.meshes.new("RING")
    
    # adding cam_body to scene
    body = bpy.data.objects.new("RING", cam_body)
    scene.collection.objects.link(body)
    
    # creating mesh of body
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=5.0)
    bm.to_mesh(cam_body)
    
    # make it rectangular
    bpy.data.objects["RING"].select_set(True)
    
    # set a specific ratio (y_scale : z_scale = 1 : 1) 
    # see if network can learn this property
    x_scale = random.uniform(0.1, 0.6)
    y_scale = 1
    z_scale = y_scale
    bpy.ops.transform.resize(value=(x_scale, y_scale, z_scale))
    #axis = random.choice(['X', 'Y', 'Z'])
    #deg = radians(random.uniform(0, 90))
    #bpy.ops.transform.rotate(value=deg, orient_axis=axis)
    
    # create a smaller cube
    cube = bpy.data.meshes.new("Cube")
    cube_scene = bpy.data.objects.new("Cube", cube)
    scene.collection.objects.link(cube_scene)
    bm2 = bmesh.new()
    bmesh.ops.create_cube(bm2, size=random.uniform(3, 4))
    bm2.to_mesh(cube)
    
    # cut out the smaller cube
    bool_one = bpy.data.objects["RING"].modifiers.new(type="BOOLEAN", name="bool 1")
    bool_one.object = bpy.data.objects["Cube"]
    bool_one.operation = 'DIFFERENCE'
    bpy.ops.object.modifier_apply(
        {"object": bpy.data.objects["RING"]}, modifier=bool_one.name)
    cube_mesh = bpy.data.meshes["Cube"]
    bpy.data.meshes.remove(cube_mesh)
    
    # triangulate the result to prevent 0 volume
    triang_one = bpy.data.objects["RING"].modifiers.new(type="TRIANGULATE", name="triang 1")
    bpy.ops.object.modifier_apply({"object": bpy.data.objects["RING"]}, modifier=triang_one.name)
    
    bpy.data.objects["RING"].select_set(False)
    
    # get faces
    bm.faces.ensure_lookup_table()
    faces = bm.faces[:]
    
def normalize():
    # select all objects (for future reference if creating more objects)
    # need to have at least one active object
    bpy.context.view_layer.objects.active = bpy.data.objects["RING"]
    for obj in bpy.data.objects:
        if obj.name.startswith("RING"):
            bpy.data.objects["RING"].select_set(True)
    bpy.ops.object.join()
    # scale each dim accordingly to restrict object in a 1x1x1 box
    max_scale = max(bpy.context.object.scale)
    bpy.context.object.scale[0] = bpy.context.object.scale[0] / max_scale
    bpy.context.object.scale[1] = bpy.context.object.scale[1] / max_scale
    bpy.context.object.scale[2] = bpy.context.object.scale[2] / max_scale
    for obj in bpy.data.objects:
        if obj.name.startswith("RING"):
            bpy.data.objects["RING"].select_set(False)

if __name__ == '__main__':
    for i in range(1000):
        print('Object #{}'.format(i + 1))
        delete_all()
        create_cam_body()
        blend_file_path = bpy.data.filepath
        dir = os.path.dirname(blend_file_path) 
        dir = os.path.join(dir, 'square_rings')
        print(dir)
        filename = str(i) + '.obj'
        target_file = os.path.join(dir, filename)
        print(target_file)
        bpy.ops.export_scene.obj(filepath=target_file)