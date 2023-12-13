
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
        self.report({'INFO'}, 'Importing ' + str(particle_positions.shape[0]) + ' particles')
        return {'FINISHED'}
    
    def add_metaballs(self,context):
        pass

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