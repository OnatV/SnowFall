import taichi as ti
from fluid_grid import FluidGrid
from numpy.random import random

# install: pip install pytest
# to execute: python -m pytest tests\test_grid.py

particle_r = 0.01
h = 4.0 * particle_r

@ti.kernel
def grid_contains(val:int, grid_idx:int, grid:ti.template()) -> bool:
    res = False
    ti.loop_config(serialize=True)
    # print("grid", grid[grid_idx])
    for i in range(grid[grid_idx].length()):
        if grid[grid_idx, i] == val:
            res = True
            break
    return res

def test_one_particle_position(num=10):
    ti.init(arch=ti.cpu, debug=True)
    origin = ti.Vector([0.0]*3)
    end_point = ti.Vector([1.0]*3)
    fg = FluidGrid(origin, end_point, h) # grid spacing is 0.12
    positions = ti.Vector.field(3, float, shape=num)
    positions.from_numpy(random((num, 3)))
    fg.update_grid(positions)
        
    @ti.kernel
    def flat_idx(pos:ti.types.vector(3, float)) -> int:
        (x,y,z) = fg.to_grid_idx(pos)
        return fg.grid_to_array_index(x, y, z)
    
    for i in range(num):
        print("position", positions[i])
        print("idx", flat_idx(positions[i]))
        assert grid_contains(i, flat_idx(positions[i]), fg.grid)

