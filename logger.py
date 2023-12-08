import taichi as ti
import numpy as np
import os
import h5py

from particle_system import ParticleSystem

class Logger:
    def __init__(self, ps: ParticleSystem,
        config,
        replay:bool
    ):
        self.ps = ps
        self.cfg = config
        self.replay = replay
        self.fields = [x.strip() for x in self.cfg.logging_fields.split(",")]
        self.log_dir = os.path.join(os.getcwd(), self.cfg.log_dir)
        self.init_logging_directory()
        print("storing log at", os.path.join(self.log_dir, 'log.hdf5'))
        self.file = None
        self.num_time_steps = 0
        self.init_log()
        self.current_step = 0 # this is an int that indexes us into the dataset
        self.log_time_step = 1.0 / self.cfg.log_fps

    def init_log(self):
        if not self.replay:
            self.file = h5py.File(os.path.join(self.log_dir, 'log.hdf5'), "w")
            self.num_time_steps = int(np.ceil(self.cfg.max_time * self.cfg.log_fps)) + 1
            print("dataset will contain", self.num_time_steps, "timesteps!")
            for field in self.fields:
                print("Creating", field, "dataset in log!")
                if field == 'position':
                    self.file.create_dataset(field, (3 * self.cfg.num_particles, self.num_time_steps), dtype='f')
                if field == 'velocity':
                    self.file.create_dataset(field, (3 * self.cfg.num_particles, self.num_time_steps), dtype='f')
                if field == 'acceleration':
                    self.file.create_dataset(field, (3 * self.cfg.num_particles, self.num_time_steps), dtype='f')
                if field == 'density':
                    self.file.create_dataset(field, (1 * self.cfg.num_particles, self.num_time_steps), dtype='f')
                if field == 'rest_density':
                    self.file.create_dataset(field, (1 * self.cfg.num_particles, self.num_time_steps), dtype='f')
                if field == 'pressure':
                    self.file.create_dataset(field, (1 * self.cfg.num_particles, self.num_time_steps), dtype='f')
        else:
            self.file = h5py.File(os.path.join(self.log_dir, 'log.hdf5'), "r")
            print("Shape of position", self.file['position'].shape)
            self.num_time_steps = int(np.ceil(self.cfg.max_time * self.cfg.log_fps)) + 1

    def init_logging_directory(self):
        os.makedirs(self.log_dir, exist_ok = True)

    def log_step(self, time):
        print("Logging time step", time)
        for field in self.fields:
            if field == 'position':
                dset = self.file['position']
                data = self.ps.position.to_numpy().reshape([3 * self.ps.num_particles])
                dset[:, self.current_step] = data
            if field == 'acceleration':
                dset = self.file['acceleration']
                data = self.ps.position.to_numpy().reshape([3 * self.ps.num_particles])
                dset[:, self.current_step] = data
            if field == 'density':
                dset = self.file['density']
                data = self.ps.position.to_numpy().reshape([self.ps.num_particles])
                dset[:, self.current_step] = data
            if field == 'rest_density':
                dset = self.file['rest_density']
                data = self.ps.position.to_numpy().reshape([self.ps.num_particles])
                dset[:, self.current_step] = data

            ## add more fields here similar to above!
        self.current_step += 1

    def replay_step(self, time):
        pos = self.file['position'][:, self.current_step].reshape([self.ps.num_particles, 3])
        self.ps.position.from_numpy(pos)
        self.current_step += 1