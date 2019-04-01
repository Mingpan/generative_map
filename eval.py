
import os
import json
import argparse
import pickle

import numpy as np

from tqdm import tqdm

from data_loader_7scenes import DataLoader
# from data_loader_robotcar import DataLoader

from model import Model
from utils import finite_diff, predict, correct
from eval_utils import VideoRecorder, plot_result, equidistant_generate, translational_error, rotational_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None,
                        help='directory to load model from. If not provided, the most recent dir'
                        ' in the default path will be used, raise an Error if failed.')
    parser.add_argument('--state_var', type=float, default=0.001,
                        help='estimated state transition variance for the model used by extended Kalman filter, '
                             'it is assumed to be independent and stay the same for all dimension')
    parser.add_argument('--visualize_dir', type=str, default='visualize',
                        help='directory to save the visualization output + video (if apply)')
    parser.add_argument('--video', default=False, action='store_true',
                        help='if a video should be recorded comparing real, reconstructed, and estimated '
                             'reconstructed images, only available for image reconstruction model')
    parser.add_argument('--no_model', default=False, action='store_true',
                        help='if the true state transition will be provided to enhance the accuracy')
    parser.add_argument('--generation_mode', type=int, default=1,
                        help='0: equidistant sample image generation; 1: Kalman filter estimation for localization; '
                             '2: perform both')
    parser.add_argument('--all', default=False, action='store_true',
                        help='evaluate all the validation sequences and report the performance w/o a picture.')
    parser.add_argument('--seq_idx', type=int, default=0,
                        help='evaluate the given indexed sequence in the text.')
    parser.add_argument('--deviation', type=float, default=0.,
                        help='deviation across all state dimension for robust test.')
    args = parser.parse_args()

    evaluate(args)


def ekf_estimate(system, model, state_var, save_video_dir=None, deviation=0.):
    """ Estimate the system with extended Kalman filter

        :param system: the system to estimate,
        :param model: the model we build for mimic the system,
        :param state_var: estimated state variance
        :param save_video_dir: if given, step wise images will be saved and recorded as a video
        :param deviation: deviation added upon each dimension of the initial state
        :returns
            x_est: estimated states by extended Kalman filter based on the images,
            Sigma_est: estimated observation uncertainties
            x_true: the recorded true system states
            o_list: list of obtained observation
    """

    system.reset()
    x = system.get_pos()
    # set initial deviation
    x = x + (np.ones_like(x) * deviation)
    model.set_norm(system.norm_xyz, system.norm_q)

    Sigma = np.eye(model.dim_state) * state_var
    Q = np.eye(model.dim_state) * state_var

    x_true = []  # true system states
    o_list = []  # observations
    x_est = []  # estimated states
    Sigma_est = []  # estimated covariances
    R_list = []  # uncertainty matrix for observations

    # init the display for video recording
    recorder = None
    if save_video_dir is not None:
        recorder = VideoRecorder(save_video_dir, norm_xyz=system.norm_xyz)

    # init observation
    img = system.render()

    print("Simulating the system with EKF...")
    for t in tqdm(range(system.horizon)):
        o, cov_o = model.wrap(img)

        # estimate observation model H
        o_func = lambda _x: model.emit(_x)
        H = finite_diff(o_func, x, dim_out=model.dim_observation)

        # get the observation variance
        R = cov_o

        # correct the state using observation and H
        x, Sigma = correct(x, Sigma, o, R, H, model.emit)

        # collect current step data
        R_list.append(np.diag(R))
        x_est.append(x)
        Sigma_est.append(Sigma)
        o_list.append(o)
        x_sys = system.get_pos()
        x_true.append(x_sys)

        if recorder is not None:
            emit = model.generate(x)
            recorder.record(img, emit, x_sys, x)

        if t < system.horizon - 1:
            # evolve the system by one step

            img, u, dt = system.step()
            dt = float(dt)

            # estimate state transition matrix A, B
            x_func = lambda _x: model.step(_x, u, dt)
            u_func = lambda _u: model.step(x, _u, dt)
            A = finite_diff(x_func, x, dim_out=model.dim_state)
            B = finite_diff(u_func, u, dim_out=model.dim_state)

            # get the transition mapping
            sim_step = lambda _x, _u: model.step(_x, _u, dt)

            # predict the next state, let Qt be correlated with dt, since longer time exhibit more uncertainty
            x, Sigma = predict(x, Sigma, u, A, B, Q*(dt**2), sim_step)

    if recorder is not None:
        recorder.finalize()

    # show the sigma of z
    print("Aver. inferred variance is {}".format(np.mean(R_list)))

    # revoke the normalization, prepare for evaluation
    x_est = np.array(x_est)
    x_est[:, :3, :] *= system.norm_xyz
    x_est[:, 3:, :] *= system.norm_q
    Sigma_est = np.array(Sigma_est)
    x_true = np.array(x_true)
    x_true[:, :3, :] *= system.norm_xyz
    x_true[:, 3:, :] *= system.norm_q
    o_list = np.array(o_list)

    return x_est, Sigma_est, x_true, o_list


def load_model_specs(model_dir):
    """Look for the model specifications in the given directory"""
    if model_dir is None:
        # get the latest modified dir in the default dir, raise an Error if there's none
        default_dir = "checkpoints"
        all_dirs = [os.path.join(default_dir, d)
                    for d in os.listdir(default_dir) if os.path.isdir(os.path.join(default_dir, d))]
        if len(all_dirs) == 0:
            raise FileNotFoundError("Model directory is not provided & no directory in default path!")
        model_dir = max(all_dirs, key=os.path.getmtime)

    if model_dir.endswith('.ckpt'):
        mdir = os.path.dirname(model_dir)
        json_path = os.path.join(mdir, "model_specs.json")
    else:
        json_path = os.path.join(model_dir, "model_specs.json")
    with open(json_path, 'r') as infile:
        data = json.load(infile)
    train_args = argparse.Namespace(**data)
    return train_args, model_dir


def evaluate(eval_args):
    train_args, model_dir = load_model_specs(eval_args.model_dir)
    if len(eval_args.visualize_dir) > 0:
        eval_args.visualize_dir = os.path.join(model_dir, eval_args.visualize_dir)
        if not os.path.isdir(eval_args.visualize_dir):
            os.makedirs(eval_args.visualize_dir)
    else:
        eval_args.visualize_dir = None

    print('Model specification loaded.')
    print(eval_args)
    print(train_args)

    model = Model(dim_observation=train_args.dim_observation,
                  batch_size=1,
                  image_size=train_args.dim_input,
                  reconstruct_size=train_args.dim_reconstruct)
    model.restore(model_dir)

    # get the dataset / simulated system
    dataloader = DataLoader(train_args.data_dir, no_model=eval_args.no_model,
                            img_width=train_args.dim_input[1], img_height=train_args.dim_input[0],
                            norm_factor=train_args.scaling_factor)

    if not eval_args.all:
        sequence = dataloader.get_valid_seq(eval_args.seq_idx)

        if eval_args.generation_mode == 0:
            equidistant_generate(sequence, model, save_path=eval_args.visualize_dir)
        else:
            video_dir = eval_args.visualize_dir if eval_args.video else None
            x_est, cov_est, x_true, ob = ekf_estimate(sequence, model, eval_args.state_var,
                                                      save_video_dir=video_dir,
                                                      deviation=eval_args.deviation)

            plot_result(x_est, x_true, save_dir=eval_args.visualize_dir)

            if eval_args.generation_mode > 1:
                equidistant_generate(sequence, model, model_traj=x_est, save_path=eval_args.visualize_dir)
    else:
        x_est_all, x_true_all, x_pred_all = [], [], []
        for i in range(len(dataloader.valid_sequences)):
            sequence = dataloader.get_valid_seq(i)
            x_est, _, x_true, _ = ekf_estimate(sequence, model, eval_args.state_var,
                                               deviation=eval_args.deviation)
            x_est_all.extend(x_est)
            x_true_all.extend(x_true)

        x_est_all = np.array(x_est_all)
        x_true_all = np.array(x_true_all)

        mean_est_err_xyz, median_est_err_xyz = translational_error(x_true_all, x_est_all)
        mean_est_err_rot, median_est_err_rot = rotational_error(x_true_all, x_est_all)
        print("Estimator:")
        print("\t Mean Error: {} m, {} deg".format(mean_est_err_xyz, mean_est_err_rot))
        print("\t Median Error: {} m, {} deg".format(median_est_err_xyz, median_est_err_rot))

        with open(os.path.join(eval_args.visualize_dir, 'evaluations.pkl'), 'wb') as f:
            data = {
                'true': x_true_all,
                'estimate': x_est_all,
                'prediction': x_pred_all
            }
            pickle.dump(data, f)


if __name__ == '__main__':
    main()

