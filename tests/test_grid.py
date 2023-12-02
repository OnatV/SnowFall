import taichi as ti
import pytest
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

@pytest.mark.parametrize("border", [(True,), (False, )])
def test_neighbors_2(border):
    '''
        tests that point 1 is in fact a neighbor of point 0
    '''
    ti.init(arch=ti.cpu, debug=True)
    origin = ti.Vector([0.0]*3)
    end_point = ti.Vector([1.0]*3)
    fg = FluidGrid(origin, end_point, 2 * h)
    positions = ti.Vector.field(3, float, shape=2)
    
    if not border:
        positions[0] = origin + 0.3
    else:
        positions[0] = origin
    positions[1] = positions[0] + (h / 2)
    fg.update_grid(positions)

    @ti.func
    def count_neighbors(i, j, sum:ti.template()):
        sum += 1

    @ti.func
    def save_neigh_index(i, j, save_arr:ti.template()):
        save_arr.append(j)

    dyn_arr = ti.root.dynamic(ti.i, 1000, chunk_size=32)
    save_arr = ti.field(int)
    dyn_arr.place(save_arr)

    @ti.kernel
    def runtest() -> bool:
        res = True
        ti.loop_config(serialize=True)
        for i in ti.grouped(positions):
            sum = 0
            fg.for_all_neighbors(i, positions, count_neighbors, sum, h)
            if sum != 1:
                print(f"sum for {i} was {sum}")
                res = False
                
                fg.for_all_neighbors(i, positions, save_neigh_index, save_arr, h)
                print(f"neighbour idx for {i} are:")
                for k in range(save_arr.length()):
                    print(save_arr[k], ", ", end="")
                print()
                save_arr.deactivate()
            
        return res
    assert runtest()
    
