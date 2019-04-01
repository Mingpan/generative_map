
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

import skimage.transform as transform

from tqdm import tqdm

from utils import quaternion_distance, inv_log_quaternion


def reverse_preprocess(img, to_int=False, shift=False):
    img = img[..., ::-1]
    if shift:
        img += 0.5
    if to_int:
        return img.astype(np.int)
    return img


class VideoRecorder:
    def __init__(self, save_dir, dim_state=3, norm_xyz=1.):
        self.dim_state = dim_state
        self.save_dir = save_dir
        self.norm_xyz = norm_xyz
        self.traj_true = np.zeros((dim_state, 0))
        self.traj_est = np.zeros((dim_state, 0))
        self.counter = 1
        self.gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 3])

    def record(self, real, emit, xt, xe):
        real = reverse_preprocess(real)
        emit = reverse_preprocess(emit)
        x_true = xt.copy()
        x_true *= self.norm_xyz
        x_est = xe.copy()
        x_est *= self.norm_xyz
        self.traj_true = np.hstack((self.traj_true, x_true[:3, :]))
        self.traj_est = np.hstack((self.traj_est, x_est[:3, :]))

        fig = plt.figure(figsize=(12, 6))
        plt.subplots_adjust(wspace=-0.2)

        ax = fig.add_subplot(self.gs[:2, :2], projection='3d')
        ax.plot(self.traj_true[0, :], self.traj_true[1, :], self.traj_true[2, :], '--b', label='true')
        ax.plot(self.traj_est[0, :], self.traj_est[1, :], self.traj_est[2, :], '-r', label='estimate')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.tick_params(pad=-0.5)
        ax.legend()

        ax = fig.add_subplot(self.gs[0, 2])
        ax.imshow(real)
        ax.set_title("Real Image")
        ax.grid(False)
        plt.axis('off')

        ax = fig.add_subplot(self.gs[1, 2])
        ax.imshow(emit)
        ax.set_title("Generated Image")
        ax.grid(False)
        plt.axis('off')

        path = os.path.join(self.save_dir, "{}.png".format(self.counter))
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        self.counter += 1

    def finalize(self):

        def update_image(num, load_dir, ax, update):
            ax.clear()
            path = os.path.join(load_dir, "{}.png".format(num+1))
            data = plt.imread(path)
            update(1)
            img = ax.imshow(data)
            plt.axis('off')
            return img,

        print("Generating video in {}...".format(self.save_dir))
        with tqdm(total=self.counter) as tbar:
            writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Me'), bitrate=1800)
            f = plt.figure(figsize=(12, 5))
            f.tight_layout()
            ax = f.add_subplot(1, 1, 1)
            line_ani = animation.FuncAnimation(f, update_image, self.counter - 1,
                                               fargs=(self.save_dir, ax, tbar.update),
                                               interval=50, blit=True)
            video_path = os.path.join(self.save_dir, "video.mp4")
            line_ani.save(video_path, writer=writer)
        plt.close(f)


def translational_error(true, pred):
    error = np.sqrt(np.sum(np.square(true[:, :3, 0] - pred[:, :3, 0]), axis=1))
    return np.mean(error), np.median(error)


def rotational_error(true, pred):
    q_true = [inv_log_quaternion(true[i, 3:, 0]) for i in range(len(true))]
    q_pred = [inv_log_quaternion(pred[i, 3:, 0]) for i in range(len(pred))]
    error = [quaternion_distance(qt, qp) for qt, qp in zip(q_true, q_pred)]
    return np.mean(error) / np.pi * 180, np.median(error) / np.pi * 180


def plot_result(x_est, x_true=None, save_dir=None):
    """Plot the output trajectories, estimate the error if possible. """
    f, axarr = plt.subplots(2, sharex='all', figsize=(12, 12))
    for p in range(2):
        start_idx = p * 3
        end_idx = min(start_idx + 3, x_est.shape[1])
        for i, idx in enumerate(range(start_idx, end_idx)):
            axarr[p].plot(x_est[:, idx, 0], '-', label='x%d' % idx, color='C%d' % i)
            if x_true is not None:
                axarr[p].plot(x_true[:, idx, 0], '--', label='x%d true' % idx, color='C%d' % i)
        axarr[p].legend()

    if x_true is not None:
        mu_xyz, median_xyz = translational_error(x_true, x_est)
        mu_rot, median_rot = rotational_error(x_true, x_est)
        axarr[0].set_title("Extended Kalman Filter Error: est = %.3fm, %.3fdeg" % (median_xyz, median_rot))
    else:
        axarr[0].set_title("Extended Kalman Filter Result")

    if save_dir is not None:
        path = os.path.join(save_dir, "filter")
        plt.savefig(path)


def equidistant_generate(system, model, num_sample=10, model_traj=None, save_path=None):
    # get system trajectory
    horizon = system.horizon
    model_traj = model_traj[:horizon, ...] if model_traj is not None else None
    system.reset()
    traj = [system.get_pos()]
    for i in tqdm(range(horizon - 1)):
        system.step()
        traj.append(system.get_pos())
    traj = np.asarray(traj)

    plotting_gap = (horizon - 1) // (num_sample - 1)
    plotting_steps = list(range(0, horizon, plotting_gap))

    real_images = []
    generated_images = []

    for i in plotting_steps:
        if model_traj is not None:
            pos = model_traj[i]
            pos[:3, :] /= system.norm_xyz
            pos[3:, :] /= system.norm_q
        else:
            pos = traj[i]
        gen = reverse_preprocess(model.generate(pos))
        real = reverse_preprocess(system.render(i))
        real = transform.resize(real, gen.shape,
                                anti_aliasing=True, mode='constant')
        real_images.append(real)
        generated_images.append(gen)
        if save_path:
            fig, ax = plt.subplots(1, frameon=False)
            ax.imshow(real)
            path = os.path.join(save_path, "real_{}".format(i))
            ax.grid(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, frameon=False)
            ax.imshow(gen)
            path = os.path.join(save_path, "generated_{}".format(i))
            ax.grid(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight')
            plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    traj[:, :3, :] *= system.norm_xyz
    traj[:, 3:, :] *= system.norm_q
    ax.plot(traj[:, 0, 0], traj[:, 1, 0], traj[:, 2, 0], '--b')
    if model_traj is not None:
        ax.plot(model_traj[:, 0, 0], model_traj[:, 1, 0], model_traj[:, 2, 0], '-r')
    for i in plotting_steps:
        ax.scatter(xs=traj[i, 0, 0], ys=traj[i, 1, 0], zs=traj[i, 2, 0], c='b', s=20)
        ax.text(traj[i, 0, 0], traj[i, 1, 0], traj[i, 2, 0], str(i), color='black')
        if model_traj is not None:
            ax.scatter(xs=model_traj[i, 0, 0], ys=model_traj[i, 1, 0], zs=model_traj[i, 2, 0], c='r', s=20)
            ax.text(model_traj[i, 0, 0], model_traj[i, 1, 0], model_traj[i, 2, 0], str(i), color='magenta')

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    ax.tick_params(pad=-0.5)

    if save_path is not None:
        path = os.path.join(save_path, "trajectory")
        plt.savefig(path, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # joint figure
    if save_path:
        shape = model.generate(traj[0]).shape
        join_figure = np.zeros((shape[0] * 2, shape[1] * num_sample, 3))
        for i, (real, gen) in enumerate(zip(real_images, generated_images)):
            s = i * shape[1]
            e = (i + 1) * shape[1]
            join_figure[0:shape[0], s:e, :] = real
            join_figure[shape[0]:2 * shape[0], s:e, :] = gen

        fig, ax = plt.subplots(1, frameon=False)
        ax.imshow(join_figure)
        path = os.path.join(save_path, "joint")
        ax.grid(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


