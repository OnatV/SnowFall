# SnowFall
- Run via
    ```
    python -m scripts.ball_and_cloth
    ```

## Current Progress

![Dev Log](images/nov_04_js.png)

for now, the snow particles can fall and collide with the floor of the domain. There are no forces calculated other than gravity.

last updated Nov 4th 2023

## Architecture:
- **main.py**: entry point for program.
    - creates/reads configuration for simulation
    - creates particle system class
    - creates solver class
- **snow_config.py**: contains configuration parameters in a neat structure that can be passed around
- **particle_system.py**: contains class definition and methods for ParticleSystem:
    - contains necessary fields for particles (position, vel, accel) as [ti.Vector.field](https://docs.taichi-lang.org/docs/field#vector-fields) types that are (N,3) arrays where N is the number of particles
    - handles visualization of the domain and particles within
- **sph_solver.py**: provides a class that contains a handle on the ParticleSystem and defines simulation methods: "implements physics"
    - contains "step" method that updates the simulation
    - contains kernels used for sph

## To-Do/Next Steps:
- implement smoothing kernel for SPH 
- calculate pressure and viscosity forces (to behave more like a fluid)

## Sources
for theory:
- https://sph-tutorial.physics-simulation.org/
- https://matthias-research.github.io/pages/publications/sca03.pdf  


for implementation inspiration:
- https://github.com/erizmr/SPH_Taichi/tree/master
- https://github.com/pmocz/sph-python/blob/master/sph.py
