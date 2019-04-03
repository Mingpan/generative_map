# The data loader for "TUM style" of dataset


import os
import numpy as np

import pickle
import skimage.transform as transform

from env import MobileRobot
from utils import log_quaternion


class DataLoader:
    def __init__(self, root_dir, img_height=96, img_width=96, dim_control=7,
                 no_model=False, norm_factor=None):
        train_datasets = ['2014-06-26-08-53-56', '2014-06-26-09-24-58']
        valid_datasets = ['2014-06-23-15-36-04', '2014-06-23-15-41-25']

        self.train_dir = [os.path.join(root_dir, path, path + '.pkl') for path in train_datasets]
        self.valid_dir = [os.path.join(root_dir, path, path + '.pkl') for path in valid_datasets]
        self.vo_dir = [os.path.join(root_dir, path, path + '_vo.pkl') for path in valid_datasets]
        self.test_dir = None

        self.height = img_height
        self.width = img_width
        self.no_model = no_model
        self.dim_control = dim_control

        # require output elements
        self.time2rgb = {}
        self.time2pos = {}

        self.train_sequences = []
        for path in self.train_dir:
            self.train_sequences.append(self.load_train(path))
        self.times_train = [t for sublist in self.train_sequences for t in sublist]

        if norm_factor is None:
            self.norm_xyz, self.norm_q = self.variance_norm()
        else:
            self.norm_xyz, self.norm_q = norm_factor

    @staticmethod
    def pos_to_state(p, norm_xyz=1., norm_q=1.):
        xyz = p[:3]
        q = p[3:]
        log_q = log_quaternion(q)
        xyz = xyz / norm_xyz
        log_q = log_q / norm_q
        return np.concatenate((xyz, log_q))

    def variance_norm(self):
        print('Estimating the translation and rotational variance...', flush=True)
        states = np.array([self.pos_to_state(self.time2pos[t]) for t in self.times_train])
        print(states.shape)
        norm_xyz = np.std(states[:, :3])
        norm_q = np.std(states[:, 3:])
        print("Std of xyz: {}; std of log q: {}".format(norm_xyz, norm_q), flush=True)
        return norm_xyz, norm_q

    def load_vo(self, vo_path):
        timestamps, vo_poses = pickle.load(open(vo_path, 'rb'))
        for ts, pose in vo_poses.items():
            pose[2] = 0.
        return timestamps, vo_poses

    def load_file(self, file_path):
        timestamps, images, poses = pickle.load(open(file_path, 'rb'))

        for ts, pose in poses.items():
            pose[2] = 0.

        return timestamps, images, poses

    def load_train(self, train_dir):
        (timestamps, images, poses) = self.load_file(train_dir)

        for ts, img in images.items():
            img = transform.resize(img, (self.height, self.width), mode='constant', anti_aliasing=True)
            images[ts] = img

        self.time2rgb.update(images)
        self.time2pos.update(poses)

        return timestamps

    def get_train_set(self, size=None, sample=True):
        """ Get random sample pairs of (inputs, labels) from training times sequence"""
        inputs, poses = [], []

        if size is None:
            size = len(self.times_train)

        # print("Collecting training data...")
        if sample:
            for i in range(size):  # tqdm(range(size)):
                rand_idx = np.random.randint(len(self.times_train))
                t = self.times_train[rand_idx]
                img = self.time2rgb[t]
                pos = self.time2pos[t]
                inputs.append(img.copy())
                poses.append(pos.copy())
        else:
            inputs = [self.time2rgb[t] for t in self.times_train]
            poses = [self.time2pos[t] for t in self.times_train]

        labels = np.array([self.pos_to_state(p, self.norm_xyz, self.norm_q) for p in poses])

        return np.array(inputs), np.array(labels)

    def get_train_seq(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.train_dir))
        time_seq, time2rgb, time2pos = self.load_file(self.train_dir[idx])

        for ts, img in time2rgb.items():
            img = transform.resize(img, (self.height, self.width), mode='constant', anti_aliasing=True)
            time2rgb[ts] = img

        seq = MobileRobot(time_seq, time2rgb, self.dim_control, time2pos, no_model=self.no_model)
        return seq

    def get_valid_seq(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.valid_dir))
        time_seq, time2rgb, time2pos = self.load_file(self.valid_dir[idx])
        _, time2vopos = self.load_vo(self.vo_dir[idx])
        # fix the nan problem due to vo dataset
        time2vopos[time_seq[-1]] = time2vopos[time_seq[-2]] = time2vopos[time_seq[-3]]

        for ts, img in time2rgb.items():
            img = transform.resize(img, (self.height, self.width), mode='constant', anti_aliasing=True)
            time2rgb[ts] = img

        seq = MobileRobot(time_seq, time2rgb, self.dim_control, time2pos, time2vopos,
                          no_model=self.no_model, norm_xyz=self.norm_xyz, norm_q=self.norm_q)
        return seq
