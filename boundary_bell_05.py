import numpy as np
import trimesh
import taichi as ti
from trimesh.transformations import rotation_matrix
from mesh_to_sdf import mesh_to_voxels, scale_to_unit_cube
from scipy.interpolate import RegularGridInterpolator
from time import time_ns
from contextlib import redirect_stdout
from numpy.linalg import norm
from fluid_grid import FluidGrid
from kernels import cubic_kernel
from pathlib import Path
from trilinear import trilinear_interpolation

ti.init(arch=ti.cpu, debug=True) # cpu, to avoid insane copying

# calculate SDF grid
box_path = Path("boundary/Box_rot.glb")
#bmw__path = Path("../bmw_1_series_m_coupe/scene_merged.glb")
mesh_path = box_path
precomp_path = mesh_path.with_name("precomp_" + mesh_path.name).with_suffix(".npy")

mesh = trimesh.load(mesh_path)
if type(mesh) is trimesh.Scene: 
    mesh = mesh.dump()[0]
mesh = scale_to_unit_cube(mesh)

if not precomp_path.exists():
    voxels, gradients = mesh_to_voxels(mesh, 48, pad=False, return_gradients=True)
    with open(precomp_path, "wb") as wf:
        np.save(wf, voxels)
        np.save(wf, gradients)
# load
with open(precomp_path, "rb") as rf:
    voxels = np.load(rf)
    gradients = np.load(rf)

transl = np.array([1, 1, 1]) * 2
mesh.apply_translation(transl)

verts = mesh.vertices 
faces = mesh.faces

voxel_scale = 1.0
x = np.linspace(-1.0, 1.0, 48) * voxel_scale + transl[0]
y = np.linspace(-1.0, 1.0, 48) * voxel_scale + transl[1]
z = np.linspace(-1.0, 1.0, 48) * voxel_scale + transl[2]
#inter_voxels = RegularGridInterpolator((x, y, z), voxels, bounds_error=False, fill_value=None )
#inter_grad   = RegularGridInterpolator((x, y, z), gradients, bounds_error=False, fill_value=None )



#num_layers = 1
particle_radius = 0.04
max_particles_per_face = 1524

# @ti.func
# def cubic_kernel(r_norm, h):
#     # implementation details borrowed from SPH_Taichi
#     # use ps.smoothing_radius to calculate the kernel weight of particles
#     # for now, sum over nearby particles
#     w = ti.cast(0.0, ti.f32)
#     k = 8 / np.pi
#     k /= ti.pow(h, 3)
#     q = r_norm / h
#     if q <= 1.0:
#         if q <= 0.5:
#             q2 = ti.pow(q, 2)
#             q3 = ti.pow(q, 3)
#             w = k * (6.0 * q3 - 6.0 * q2 + 1)
#         else:
#             w = k * 2 * ti.pow(1 - q, 3.0)
#     return w

# a class to hold all the sim data
@ti.data_oriented
class Data:
    def __init__(self, mesh_vertices, mesh_faces) -> None:
        self.voxel_axis = ti.field(float, shape=48)
        self.voxel_axis.from_numpy(np.linspace(-1.0, 1.0, 48) + transl[0])
        
        self.sdf = ti.field(float, shape=voxels.shape)
        self.sdf.from_numpy(voxels)
        self.sdf_grad = ti.Vector.field(3, float, shape=gradients.shape[:-1])
        self.sdf_grad.from_numpy(gradients)
        numFaces = mesh_faces.shape[0]
        self.time_step = 0.1
        self.h = particle_radius * 2.1
        self.origin = ti.Vector([0.0]*3)
        self.grid_end = ti.Vector([6.0]*3)
        self.grid_spacing = self.h * 1.2
        self.fg = FluidGrid(self.origin, self.grid_end, self.grid_spacing)
        print("grid size:", self.fg.num_cells)
        print("num faces:", numFaces)
        self.particle_nums = np.empty(dtype=int, shape=numFaces)
        tmp_pos = []
        self.allow_internal_particles = True
        #self.layer_offsets = np.linspace(0.0, -1.0, num_layers)
        particle_sum = 0
        # here we init particles per face
        for tri_idx in range(numFaces):
            print("\r", tri_idx, end="")
            face = mesh_faces[tri_idx]
            x0 = mesh_vertices[face[0]]
            x1 = mesh_vertices[face[1]]
            x2 = mesh_vertices[face[2]]
            e1 = x1 - x0
            e2 = x2 - x0

            # calculate D*A / (pi*r^2)
            sample_density = 1.7
            area = norm(np.cross(e1, e2)) / 2.0
            numParticles = sample_density * area / (np.pi * particle_radius**2)
            numParticles = int(numParticles) + \
                (1 if np.random.uniform() < (numParticles - int(numParticles)) else 0)
            numParticles = min(numParticles, max_particles_per_face)
            pos = np.empty(shape=(numParticles, 3), dtype=np.float32)

            rand = np.random.uniform(size= 2 * numParticles).reshape((numParticles, 2))
            
            for i in range(numParticles):
                x = e1 * rand[i][0] + (1-rand[i][0])*rand[i][1]*e2
                pos[i] = ti.Vector(x+x0)
            self.particle_nums[tri_idx] = numParticles
            particle_sum += numParticles
            tmp_pos.append(pos)
        print()
        print("num particles", particle_sum)
        self.pos = ti.Vector.field(n=3, dtype=float, shape=particle_sum)
        self.vel = ti.Vector.field(n=3, dtype=float, shape=particle_sum)
        self.colors = ti.Vector.field(n=3, dtype=float, shape=particle_sum)
        colors = [
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 1.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector([1.0, 0.0, 1.0]),
            ti.Vector([1.0, 1.0, 1.0]),
            ti.Vector([0.0, 1.0, 1.0]),
        ]
        # write all the positions into a single ti.field
        tmp_pos = np.concatenate(tmp_pos)
        self.pos.from_numpy(tmp_pos)
        self.fg.update_grid(self.pos)
        self.color_density()

    @ti.kernel
    def update_positions(self):
        for i in ti.grouped(self.vel):
            self.pos[i] += self.time_step * self.vel[i]
            self.pos[i] = ti.math.clamp(self.pos[i], self.origin, self.grid_end)
            
    @ti.func
    def aux_update_velocities(self, i, j, vr:ti.template()):        
        if i[0] != j:
            r = (self.pos[i] - self.pos[j])
            vr += cubic_kernel(r.norm(), self.h) * r.normalized()
            #if i[0] % 300 == 0:
            #    print(vr, end="")

    @ti.kernel
    def update_vr(self): # V_r part
        for i in ti.grouped(self.vel):
            self.vel[i] = [0,0,0]
        
        for i in ti.grouped(self.vel):
            self.fg.for_all_neighbors(i, self.pos, self.aux_update_velocities, self.vel[i], self.h)
            vel_norm = self.vel[i].normalized(0.0001)
            
            self.vel[i] = vel_norm * self.h
            #if vr.norm() > self.h * 2:
            #    vr = vr / vr.norm() * self.h * 2
        
        
            
    @ti.func
    def max_velocity(self) -> float:
        ret = ti.Vector([-1.0], float)
        for i in range(self.vel.shape[0]):
            ti.atomic_max(ret, self.vel[i].norm())
        return ret[0]

    @ti.kernel
    def scale_velocity(self):
        """scale velocities be no more than 1 in norm"""
        maxvel = self.max_velocity()
        for i in range(self.vel.shape[0]):
            self.vel[i] /= maxvel
    @ti.func
    def colorize(self, i, j, ret:ti.template()):
        if i == j:
            self.colors[j] = [0.0, 0.0, 1.0]
        else:
            self.colors[j] = [1.0, 0.0, 0.0]

    @ti.kernel
    def neigh_color(self, i:int):
        self.fg.for_all_neighbors(i, self.pos, self.colorize, [], self.h)

    @ti.kernel
    def color_density(self):
        red = [1, 0, 0]
        white = [1, 1, 1]
        max_density = 500
        for i in range(self.colors.shape[0]):
            density = ti.Vector([0], float)
            self.fg.for_all_neighbors(i, self.pos, self.calc_density, density, self.h)
            density_ratio = ti.math.clamp(density, 0, max_density) / max_density
            self.colors[i] = red * density_ratio[0] + (1 - density_ratio[0]) * white

    @ti.func
    def calc_density(self, i, j, ret:ti.template()):
        rnorm = (self.pos[i] - self.pos[j]).norm()
        ret += 0.01 * cubic_kernel(rnorm, self.h)

    def update_velocity(self):
        self.update_vr()
        self.update_vf()

    @ti.kernel
    def update_vf(self):
        for i in range(self.pos.shape[0]):
            p = self.pos[i]

            phi = trilinear_interpolation(self.sdf, self.voxel_axis, self.voxel_axis, self.voxel_axis, p)
            n = trilinear_interpolation(self.sdf_grad, self.voxel_axis, self.voxel_axis, self.voxel_axis, p)
            # test: make v_r purely parallel to surface
            #tmp = n.dot(self.vel[i])
            #self.vel[i] -= n*tmp
            if ti.static(self.allow_internal_particles):
                phi = max(0, phi)

            v_f = -phi * n
            self.vel[i] += v_f * 7.0

data = Data(verts, faces)

vertices = ti.Vector.field(n=3, dtype=float, shape=verts.shape[0])
vertices.from_numpy(verts)
indices = ti.field(int, shape=faces.shape[0] * 3)
indices.from_numpy(faces.flatten())

window = ti.ui.Window("display", res=(800,800), vsync=True)
camera = ti.ui.Camera()
camera.up(0.0, 1.0, 0.0)
camera.position(0, 1.5, 0.5)
camera.lookat(1, 1.5, 1 )
canvas = window.get_canvas()
scene = ti.ui.Scene()

particle_sim = False

iteration = 0
while window.running:
    if window.is_pressed(ti.ui.SPACE, ' '): particle_sim = True
    if window.is_pressed(ti.ui.ALT): particle_sim = False
    if window.is_pressed(ti.ui.BACKSPACE):
        window.running = False
    # if window.is_pressed(ti.ui.UP):
    #     data.pos[0].x += 0.05
    # if window.is_pressed(ti.ui.DOWN):
    #     data.pos[0].x -= 0.05
    # if window.is_pressed(ti.ui.LEFT):
    #     data.pos[0].z += 0.05
    # if window.is_pressed(ti.ui.RIGHT):
    #     data.pos[0].z -= 0.05
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.mesh(vertices=vertices, indices=indices ,color=(1.0,0.0,0.0), show_wireframe=True)
    #data.color_reset()
    #data.neigh_color(0)
    if particle_sim:   
        #ti.profiler.clear_kernel_profiler_info()
        
        ta = time_ns() 
        data.update_velocity()
        te = time_ns()
        #print(f"\rvel step: {(te - ta) / 1e6} ms", end="")
        data.update_positions()
        data.fg.update_grid(data.pos)
        data.color_density()
        
    scene.particles(data.pos, radius=particle_radius*1.0, per_vertex_color=data.colors)
    canvas.scene(scene)

    window.show()
    data.time_step = 0.18 / (iteration + 1) ** 0.2
    iteration = min(iteration+1, 100)

output_path = precomp_path.with_name("output_" + mesh_path.name).with_suffix(".npy")
pos_np = data.pos.to_numpy()
pos_np -= transl
with open(output_path, "wb") as wf:
    np.save(wf, pos_np)
