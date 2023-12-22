
bl_info ={
    "name":"SnowFall",
    "description":"Import simulated snow",
    "author":"Jackson Stanhope",
    "version":(0,0,0),
    "blender":(3,5,0),
    "location":"View 3D > Properties Panel",
    "support":"COMMUNITY",
    "category":"Object",
}

## the material was borrowed from Blender's official "RealSnow" addon

import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector
import math
import os
import numpy as np

class SnowFall_PANEL(Panel):
    bl_space_type = "VIEW_3D"
    bl_context = "objectmode"
    bl_region_type = "UI"
    bl_label = "SnowFall"
    bl_category = "SnowFall"

    def draw(self, context):
        scn = context.scene
        settings = scn.snowfall
        layout = self.layout
        col = layout.column(align=True)
        col.prop(settings, 'import_path')
        col.prop(settings, 'radius')
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("snowfall.import", text="Import Snow")

class SnowFallSettings(PropertyGroup):
    import_path : StringProperty(
        name = "Import Path",
        description="File path of exported simulation",
        default="",
        subtype='FILE_PATH'
    )
    radius : FloatProperty(
        name = "Radius",
        description = "Radius of Snow Particles",
        default = 0.02,
    )

class SnowFall_Import(Operator):
    bl_idname = "snowfall.import"
    bl_label = "Import Snow"
    bl_description = "Import snow"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        print("Hello World")
        import_path = bpy.path.abspath(context.scene.snowfall.import_path)
        self.report({'INFO'}, 'Importing snow from ' + import_path)
        # load particle positions
        particle_positions = np.load(import_path)
        
        #create metaball
        ball = add_metaball(context)
        
        # create mesh
        me = bpy.data.meshes.new("SnowMesh")
        ob = bpy.data.objects.new("SnowMesh", me)
        context.collection.objects.link(ob)
        
        bm = bmesh.new()
        
        for pos in particle_positions:
            bm.verts.new(pos)
            
        bm.to_mesh(me)
        bm.free()
        
        objects = bpy.data.objects
        snow_obj = objects["SnowMesh"]
        ball.parent = snow_obj
        snow_obj.instance_type = 'VERTS'
        add_material(snow_obj)
        self.report({'INFO'}, 'Importing ' + str(particle_positions.shape[0]) + ' particles')
        return {'FINISHED'}
    
def add_metaball(context):
    ball_name = "MetaBall"
    ball = bpy.data.metaballs.new(ball_name)
    ballobj = bpy.data.objects.new(ball_name, ball)
    bpy.context.scene.collection.objects.link(ballobj)
    ball.resolution = 0.0001
    ball.threshold = 1.3
    element = ball.elements.new()
    element.radius = 1.5 * context.scene.snowfall.radius
    element.stiffness = 0.75
    ballobj.scale = [1.0, 1.0, 1.0]
    return ballobj

def add_material(obj: bpy.types.Object):
    mat_name = "Snow"
    # If material doesn't exist, create it
    if mat_name in bpy.data.materials:
        bpy.data.materials[mat_name].name = mat_name+".001"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # Delete all nodes
    for node in nodes:
        nodes.remove(node)
    # Add nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    vec_math = nodes.new('ShaderNodeVectorMath')
    com_xyz = nodes.new('ShaderNodeCombineXYZ')
    dis = nodes.new('ShaderNodeDisplacement')
    mul1 = nodes.new('ShaderNodeMath')
    add1 = nodes.new('ShaderNodeMath')
    add2 = nodes.new('ShaderNodeMath')
    mul2 = nodes.new('ShaderNodeMath')
    mul3 = nodes.new('ShaderNodeMath')
    range1 = nodes.new('ShaderNodeMapRange')
    range2 = nodes.new('ShaderNodeMapRange')
    range3 = nodes.new('ShaderNodeMapRange')
    vor = nodes.new('ShaderNodeTexVoronoi')
    noise1 = nodes.new('ShaderNodeTexNoise')
    noise2 = nodes.new('ShaderNodeTexNoise')
    noise3 = nodes.new('ShaderNodeTexNoise')
    mapping = nodes.new('ShaderNodeMapping')
    coord = nodes.new('ShaderNodeTexCoord')
    # Change location
    output.location = (100, 0)
    principled.location = (-200, 600)
    vec_math.location = (-400, 400)
    com_xyz.location = (-600, 400)
    dis.location = (-200, -100)
    mul1.location = (-400, -100)
    add1.location = (-600, -100)
    add2.location = (-800, -100)
    mul2.location = (-1000, -100)
    mul3.location = (-1000, -300)
    range1.location = (-400, 200)
    range2.location = (-1200, -300)
    range3.location = (-800, -300)
    vor.location = (-1500, 200)
    noise1.location = (-1500, 0)
    noise2.location = (-1500, -250)
    noise3.location = (-1500, -500)
    mapping.location = (-1700, 0)
    coord.location = (-1900, 0)
    # Change node parameters
    principled.distribution = "MULTI_GGX"
    principled.subsurface_method = "RANDOM_WALK"
    principled.inputs[0].default_value[0] = 0.904
    principled.inputs[0].default_value[1] = 0.904
    principled.inputs[0].default_value[2] = 0.904
    principled.inputs[1].default_value = 1
    principled.inputs[2].default_value[0] = 0.36
    principled.inputs[2].default_value[1] = 0.46
    principled.inputs[2].default_value[2] = 0.6
    principled.inputs[3].default_value[0] = 0.904
    principled.inputs[3].default_value[1] = 0.904
    principled.inputs[3].default_value[2] = 0.904
    principled.inputs[7].default_value = 0.224
    principled.inputs[9].default_value = 0.1
    principled.inputs[15].default_value = 0.1
    vec_math.operation = "MULTIPLY"
    vec_math.inputs[1].default_value[0] = 0.5
    vec_math.inputs[1].default_value[1] = 0.5
    vec_math.inputs[1].default_value[2] = 0.5
    com_xyz.inputs[0].default_value = 0.36
    com_xyz.inputs[1].default_value = 0.46
    com_xyz.inputs[2].default_value = 0.6
    dis.inputs[1].default_value = 0.1
    dis.inputs[2].default_value = 0.3
    mul1.operation = "MULTIPLY"
    mul1.inputs[1].default_value = 0.1
    mul2.operation = "MULTIPLY"
    mul2.inputs[1].default_value = 0.6
    mul3.operation = "MULTIPLY"
    mul3.inputs[1].default_value = 0.4
    range1.inputs[1].default_value = 0.525
    range1.inputs[2].default_value = 0.58
    range2.inputs[1].default_value = 0.069
    range2.inputs[2].default_value = 0.757
    range3.inputs[1].default_value = 0.069
    range3.inputs[2].default_value = 0.757
    vor.feature = "N_SPHERE_RADIUS"
    vor.inputs[2].default_value = 30
    noise1.inputs[2].default_value = 12
    noise2.inputs[2].default_value = 2
    noise2.inputs[3].default_value = 4
    noise3.inputs[2].default_value = 1
    noise3.inputs[3].default_value = 4
    mapping.inputs[3].default_value[0] = 12
    mapping.inputs[3].default_value[1] = 12
    mapping.inputs[3].default_value[2] = 12
    # Link nodes
    link = mat.node_tree.links
    link.new(principled.outputs[0], output.inputs[0])
    link.new(vec_math.outputs[0], principled.inputs[2])
    link.new(com_xyz.outputs[0], vec_math.inputs[0])
    link.new(dis.outputs[0], output.inputs[2])
    link.new(mul1.outputs[0], dis.inputs[0])
    link.new(add1.outputs[0], mul1.inputs[0])
    link.new(add2.outputs[0], add1.inputs[0])
    link.new(mul2.outputs[0], add2.inputs[0])
    link.new(mul3.outputs[0], add2.inputs[1])
    link.new(range1.outputs[0], principled.inputs[14])
    link.new(range2.outputs[0], mul3.inputs[0])
    link.new(range3.outputs[0], add1.inputs[1])
    link.new(vor.outputs[4], range1.inputs[0])
    link.new(noise1.outputs[0], mul2.inputs[0])
    link.new(noise2.outputs[0], range2.inputs[0])
    link.new(noise3.outputs[0], range3.inputs[0])
    link.new(mapping.outputs[0], vor.inputs[0])
    link.new(mapping.outputs[0], noise1.inputs[0])
    link.new(mapping.outputs[0], noise2.inputs[0])
    link.new(mapping.outputs[0], noise3.inputs[0])
    link.new(coord.outputs[3], mapping.inputs[0])
    # Set displacement and add material
    mat.cycles.displacement_method = "DISPLACEMENT"
    obj.data.materials.append(mat)

classes = (SnowFall_PANEL, SnowFall_Import, SnowFallSettings)

register, unregister = bpy.utils.register_classes_factory(classes)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.snowfall = PointerProperty(type=SnowFallSettings)

# Unregister
def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.snowfall
if __name__ == "__main__":
    register()