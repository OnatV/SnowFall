import numpy as np
import taichi as ti

@ti.func
def find_cube_base_index(
        xAxis:ti.template(),
        yAxis:ti.template(),
        zAxis:ti.template(),
        point:ti.types.vector(3, float)):
    nAxis = xAxis.shape[0] - 1
    ax_len = xAxis[xAxis.shape[0] - 1] - xAxis[0]
    x_0 = ti.math.floor((point[0] - xAxis[0]) / ax_len * nAxis)
    y_0 = ti.math.floor((point[1] - yAxis[0]) / ax_len * nAxis)
    z_0 = ti.math.floor((point[2] - zAxis[0]) / ax_len * nAxis)
    return ti.Vector([x_0, y_0, z_0], int)

# thanks wikipedia
@ti.func
def trilinear_interpolation(
        voxels:ti.template(),
        xAxis:ti.template(),
        yAxis:ti.template(),
        zAxis:ti.template(),
        point:ti.types.vector(3, float)):
    last_idx = int(xAxis.shape[0] - 1)
    
    point = ti.math.clamp(
        point,
        ti.Vector([xAxis[0], yAxis[0], zAxis[0]], float),
        ti.Vector([xAxis[last_idx], yAxis[last_idx], zAxis[last_idx]], float))
    base_idx = find_cube_base_index(xAxis, yAxis, zAxis, point)
    base_idx = ti.math.clamp(
        base_idx, [0,0,0],
        [xAxis.shape[0] - 2, yAxis.shape[0] - 2, zAxis.shape[0] - 2])
    xi, yi, zi = base_idx[0], base_idx[1], base_idx[2]
    #print(f"xi {xi} yi {yi}, zi {zi}")
    x_d = (point[0] - xAxis[xi]) / (xAxis[xi+1] - xAxis[xi])
    y_d = (point[1] - yAxis[yi]) / (yAxis[yi+1] - yAxis[yi])
    z_d = (point[2] - zAxis[zi]) / (zAxis[zi+1] - zAxis[zi])
    #print(f"xd {x_d} yd {y_d}, zd {z_d}")

    c_000 = voxels[base_idx]
    c_001 = voxels[[xi, yi, zi+1]]
    c_010 = voxels[[xi, yi+1, zi]]
    c_011 = voxels[[xi, yi+1, zi+1]]
    c_100 = voxels[[xi+1, yi, zi]]
    c_101 = voxels[[xi+1, yi, zi+1]]
    c_110 = voxels[[xi+1, yi+1, zi]]
    c_111 = voxels[[xi+1, yi+1, zi+1]]
    
    c_00 = c_000 * (1 - x_d) + c_100 * x_d
    c_01 = c_001 * (1 - x_d) + c_101 * x_d
    c_10 = c_010 * (1 - x_d) + c_110 * x_d
    c_11 = c_011 * (1 - x_d) + c_111 * x_d
    c_0 =  c_00 * (1 - y_d) + c_10 * y_d
    c_1 =  c_01 * (1 - y_d) + c_11 * y_d
    c = c_0 * (1 - z_d) + c_1 * z_d
    return c


if __name__ == "__main__":
    
    ti.init(arch=ti.cpu, debug=True)

    grid = ti.field(float, shape=(10,)*3)
    xAxis = ti.field(float, shape=10)
    yAxis = ti.field(float, shape=10)
    zAxis = ti.field(float, shape=10)
    a = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    xAxis.from_numpy(a)
    yAxis.from_numpy(a)
    zAxis.from_numpy(a)
    @ti.kernel
    def test():
        for I in ti.grouped(grid):
            grid[I] = I[0] * I[1] * I[2]
        res = trilinear_interpolation(grid, xAxis, yAxis, zAxis, ti.Vector([-2, -10, 50]))
        print(res)
        
    test()