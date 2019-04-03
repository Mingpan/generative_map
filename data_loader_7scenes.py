# The data loader for "TUM style" of dataset


import os
import numpy as np
import random
from PIL import Image
import re

from tqdm import tqdm

from env import MobileRobot
from utils import rotation_matrix_to_quaternion, log_quaternion
from glob import glob


class CallableDict(dict):
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if callable(val):
            return val(key)
        return val


class DataLoader:
    def __init__(self, root_dir, img_height=96, img_width=96, header_lines=0, dim_control=7,
                 norm_factor=None, no_model=False):

        split_file = os.path.join(root_dir, "TrainSplit.txt")
        with open(split_file, 'r') as f:
            data = [re.findall(r'\d+', line)[0] for line in f]
        train_path = []
        for idx in data:
            if len(idx) < 2:
                idx = '0' + idx
            train_path.append('seq-' + idx)

        split_file = os.path.join(root_dir, "TestSplit.txt")
        with open(split_file, 'r') as f:
            data = [re.findall(r'\d+', line)[0] for line in f]
        test_path = []
        for idx in data:
            if len(idx) < 2:
                idx = '0' + idx
            test_path.append('seq-' + idx)

        self.train_dir = [os.path.join(root_dir, path) for path in train_path]
        self.valid_dir = [os.path.join(root_dir, path) for path in test_path]
        self.test_dir = [os.path.join(root_dir, path) for path in test_path]
        
        self.height = img_height
        self.width = img_width
        self.header_lines = header_lines
        self.no_model = no_model
        self.dim_control = dim_control

        # require output elements
        self.time2rgb = CallableDict()
        self.time2rgb_unknown = CallableDict()
        self.time2pos = {}

        self.last_timestamp = 0
        
        self.train_sequences = []
        for path in self.train_dir:
            self.train_sequences.append(self.load_train(path))
        self.times_train = [t for sublist in self.train_sequences for t in sublist]

        if norm_factor is None:
            self.norm_xyz, self.norm_q = self.variance_norm()
        else:
            self.norm_xyz, self.norm_q = norm_factor
        
        self.valid_sequences = []
        for path in self.valid_dir:
            self.valid_sequences.append(self.load_train(path))
        self.times_valid = [t for sublist in self.valid_sequences for t in sublist]
        
        self.test_sequences = []
        for path in self.test_dir:
            self.test_sequences.append(self.load_test(path))
        self.times_test = [t for sublist in self.test_sequences for t in sublist]

    def load_rgb(self, rgb_path):
        img = Image.open(rgb_path)
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        data = np.array(img.getdata()).reshape(self.height, self.width, 3)
        data = data[..., ::-1].astype(np.float32) / 255.
        return data

    def load_train(self, train_dir):
        timestamps = []
        time2rbgpath = {}
        
        # load images
        image_list = sorted(glob(os.path.join(train_dir, "*.color.png")))
        for i, file_path in enumerate(image_list):
            timestamp = i + self.last_timestamp
            timestamps.append(timestamp)
            time2rbgpath[timestamp] = file_path
            self.time2rgb[timestamp] = lambda _timestamp: self.load_rgb(time2rbgpath[_timestamp])
        
        # load states
        pose_list = sorted(glob(os.path.join(train_dir, "*.pose.txt")))
        for i, truth_path in enumerate(pose_list):
            timestamp = i + self.last_timestamp
            with open(truth_path, 'r') as f_pos:
                # get transformation matrix H
                H = [list(map(float, line.strip().split())) for line in f_pos]
                H = np.array(H)
                # get euler angles
                R = H[:-1, :-1]
                # angles = rotation_matrix_to_euler_angle(R)
                q = rotation_matrix_to_quaternion(R)
                # get position
                xyz = H[:-1, -1]
                new_pos = np.concatenate((xyz, q))
                # store it into the dict
                self.time2pos[timestamp] = new_pos
        
        self.last_timestamp = timestamp + 1
        
        return timestamps

    def load_test(self, test_dir):
        timestamps = []
        time2rbgpath = {}
        
        # load images
        image_list = sorted(glob(os.path.join(test_dir, "*.color.png")))
        for i, file_path in enumerate(image_list):
            timestamp = i + self.last_timestamp
            timestamps.append(timestamp)
            time2rbgpath[timestamp] = file_path
            self.time2rgb_unknown[timestamp] = lambda _timestamp: self.load_rgb(time2rbgpath[_timestamp])

        self.last_timestamp = timestamp + 1
        
        return timestamps

    def get_train_set(self, size=None, sample=True):
        """ Get random sample pairs of (inputs, labels) from training times sequence"""
        inputs, labels = [], []

        if size is None:
            size = len(self.times_train)

        def pos_to_state(p):
            xyz = p[:3]
            q = p[3:]
            log_q = log_quaternion(q)
            return np.concatenate((xyz, log_q))

        # print("Collecting training data...")
        if sample:
            for i in range(size):  # tqdm(range(size)):
                rand_idx = np.random.randint(len(self.times_train))
                t = self.times_train[rand_idx]
                img = self.time2rgb[t]
                pos = self.time2pos[t]
                lab = pos_to_state(pos)
                inputs.append(img.copy())
                labels.append(lab.copy())
        else:
            inputs = [self.time2rgb[t] for t in self.times_train]
            labels = [pos_to_state(self.time2pos[t]) for t in self.times_train]

        # normalize
        labels = np.array(labels)
        labels[:, :3] /= self.norm_xyz
        labels[:, 3:] /= self.norm_q

        return np.array(inputs), labels

    def get_train_seq(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.train_sequences))
        time_seq = self.train_sequences[idx].copy()
        seq = MobileRobot(time_seq, self.time2rgb, self.dim_control, self.time2pos, no_model=self.no_model)
        return seq

    def get_valid_seq(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.valid_sequences))
        time_seq = self.valid_sequences[idx].copy()
        seq = MobileRobot(time_seq, self.time2rgb, self.dim_control, self.time2pos,
                          no_model=self.no_model, norm_xyz=self.norm_xyz, norm_q=self.norm_q)
        return seq

    def get_test_seq(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.test_sequences))
        time_seq = self.test_sequences[idx].copy()
        seq = MobileRobot(time_seq, self.time2rgb_unknown, self.dim_control)
        return seq

    def variance_norm(self):
        print('Estimating the translation and rotational variance...', flush=True)
        poses = []
        for idx in tqdm(range(len(self.train_sequences))):
            seq = self.get_train_seq(idx)
            poses.append(seq.get_pos())
            for i in range(seq.horizon-1):
                seq.step()
                p = seq.get_pos()
                poses.append(p)
        poses = np.array(poses)[:, :, 0]
        norm_xyz = np.std(poses[:, :3])
        norm_q = np.std(poses[:, 3:])
        print("Std of xyz: {}; std of log q: {}".format(norm_xyz, norm_q), flush=True)
        return norm_xyz, norm_q
