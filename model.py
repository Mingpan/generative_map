
import os
import numpy as np
import skimage.transform as transform

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import quaternion_multiply, inv_log_quaternion, log_quaternion

from gen_map import GenMap

plt.style.use("seaborn")


class Model:
    def __init__(self, batch_size=100, dim_state=6, dim_control=7, dim_observation=2,
                 image_size=(96, 96, 3), reconstruct_size=(64, 64, 3), reconstruction_accuracy=0.,
                 training=False, learning_rate=1e-3):
        self.dim_state = dim_state
        self.dim_control = dim_control
        self.dim_observation = dim_observation
        self.batch_size = batch_size
        self.image_size_origin = tuple(image_size)
        self.reconstruct_size_origin = tuple(reconstruct_size)
        self.reconstruct_size = np.prod(reconstruct_size)
        self.image_size = np.prod(image_size)
        self.norm_xyz = self.norm_q = 1.

        self.network = GenMap(arch_encoder_attribute=[self.dim_state, 512, self.dim_observation],
                              image_size=image_size,
                              dim_features=512,
                              num_conv_layer=4,
                              batch_size=batch_size,
                              rec_log_var=reconstruction_accuracy,
                              training=training,
                              reconstruct_size=self.reconstruct_size_origin,
                              learning_rate=learning_rate)

    def set_norm(self, norm_xyz, norm_q):
        self.norm_xyz = norm_xyz
        self.norm_q = norm_q

    def step(self, x, u, dt):
        """Apply quaternion for step"""
        xyz = (x[:3, 0] * self.norm_xyz + u[:3, 0] * dt) / self.norm_xyz
        qt = quaternion_multiply(u[3:, 0], inv_log_quaternion(x[3:, 0] * self.norm_q))
        x1 = np.concatenate((xyz, log_quaternion(qt) / self.norm_q), axis=0).reshape(-1, 1)
        return x1

    def emit(self, state):
        state = state.reshape(-1, self.dim_state)
        observation = self.network.infer_by_x(state)
        observation = observation.reshape(self.dim_observation, -1)
        return observation

    def generate(self, state):
        state = state.reshape(-1, self.dim_state)
        img = self.network.construct_y_from_x(state)
        img = img.reshape(self.reconstruct_size_origin)
        img = np.clip(img, 0, 1.)
        return img

    def reconstruct(self, img):
        img = img.reshape((-1,) + self.image_size_origin)
        rec = self.network.reconstruct(img)
        rec = rec.reshape(self.reconstruct_size_origin)
        rec = np.clip(rec, 0, 1.)
        return rec

    def wrap(self, img):
        img = img.reshape((-1,) + self.image_size_origin)  # for CNN
        observation = self.network.infer_by_y(img)
        cov_observation = self.network.cov_infer_by_y(img)
        observation = observation.reshape(self.dim_observation, -1)
        return observation, cov_observation

    def train(self, data_loader, model_dir, save_every=5, epoch=100):

        def update(loss_list, loss_list1, loss_list2, ep):
            """
            Plot the intemediate training loss curve, save it to the model dir, and save the checkpoint.
            :param loss_list: overall loss
            :param loss_list1: reconstruction loss
            :param loss_list2: KL-divergence loss
            :param ep: current epoch index
            :return: None
            """
            f = plt.figure()
            plt.plot(loss_list, label='overall')
            plt.plot(loss_list1, label='reconstruction')
            plt.plot(loss_list2, label='kl_div')
            plt.title("training curve")
            plt.xlabel("number of batch")
            plt.ylabel("-ELBO")
            p = int(0.05 * len(loss_list))
            _max = np.max([loss_list[p], loss_list1[p], loss_list2[p]])
            _min = np.min([loss_list, loss_list1, loss_list2])
            _range = _max - _min
            print("Error: {}, Reconstruction: {}, KL-div: {}".format(np.mean(loss_list[-100:]),
                                                                     np.mean(loss_list1[-100:]),
                                                                     np.mean(loss_list2[-100:])))
            plt.ylim([_min - 0.03 * _range, _max + _range])
            plt.legend()
            path = os.path.join(model_dir, "training_curve.png")
            plt.savefig(path)
            checkpoint_path = os.path.join(model_dir, "model_{}.ckpt".format(ep))
            self.save(checkpoint_path)
            plt.close(f)

        def get_epoch(dataset):
            train_inputs, train_targets = dataset.get_train_set(sample=False)
            reconstructs = []
            for im in train_inputs:
                reconstructs.append(transform.resize(im, self.reconstruct_size_origin,
                                                     anti_aliasing=True, mode='constant').reshape(-1))  # * 255.)
            reconstructs = np.array(reconstructs)
            return train_inputs, train_targets, reconstructs

        loss_all, loss_all1, loss_all2 = [], [], []
        input_len = len(data_loader.times_train)

        inputs, targets, reconstructs = get_epoch(data_loader)

        print("Training the observer...")
        with tqdm(total=(epoch * input_len)) as tbar:
            for e in range(epoch):
                loss_list, loss_list1, loss_list2 = [], [], []
                num_iters = int(input_len / self.batch_size)
                if num_iters * self.batch_size < input_len:
                    num_iters += 1

                for i in range(num_iters):
                    idx = np.random.choice(input_len, size=self.batch_size)
                    batch_inputs = inputs[idx, ...]
                    batch_targets = targets[idx, ...]
                    batch_reconstructs = reconstructs[idx, ...]

                    loss, loss_1, loss_2 = self.network.batch_train(batch_targets, batch_inputs, batch_reconstructs)
                    loss_list.append(loss)
                    loss_list1.append(loss_1)
                    loss_list2.append(loss_2)

                final_mse = np.mean(loss_list)
                tbar.set_postfix(err="%.3f" % final_mse)
                tbar.update(input_len)
                loss_all.extend(loss_list)
                loss_all1.extend(loss_list1)
                loss_all2.extend(loss_list2)
                if (e+1) % save_every == 0:
                    update(loss_all, loss_all1, loss_all2, e + 1)
        # save the final model
        checkpoint_path = os.path.join(model_dir, 'model.ckpt')
        self.save(checkpoint_path)
        return loss_all

    def save(self, path):
        self.network.save(path)

    def restore(self, path):
        self.network.restore(path)

    def close(self):
        self.network.destruct()
