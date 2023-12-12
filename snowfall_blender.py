
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
import sys
sys.path.append('C:/Users/jacks/AppData/Local/Programs/Python/Python311/Lib/site-packages/h5py/')
import h5py

import bpy
import bmesh
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector
import math
import os

class SnowFall_PANEL(Panel):
    bl_space_type = "VIEW_3D"
    bl_context = "objectmode"
    bl_region_type = "UI"
    bl_label = "Snow"
    bl_category = "SnowFall"

    def draw(self, context):
        scn = context.scene
        settings = scn.snowfall
        layout = self.layout
        col = layout.column(align=True)
        col.prop(settings, 'import_path')

class SnowFallSettings(PropertyGroup):
    import_path : StringProperty(
        name = "Import Path",
        description="File path of exported simulation",
        default="",
        subtype='DIR_PATH'
    )

# class SnowFall_Import(Operator):
#     bl_sp

classes = (SnowFall_PANEL, SnowFallSettings)

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