
import os
import time
import json
import argparse

from data_loader_7scenes import DataLoader
# from data_loader_robotcar import DataLoader

from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_size', type=int, default=3000,
                        help='size of the training set, sampled from the given sequences')
    parser.add_argument('--dim_observation', type=int, default=128,
                        help='dimension of the latent representation ("observation")')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='mini-batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='number of epochs for training')
    parser.add_argument('--save_every', type=int, default=200,
                        help='intermediate saving frequency by epochs')
    parser.add_argument('--model_dir', type=str, default="checkpoints/%s" % time.strftime("%Y-%m-%d_%H-%M-%S"),
                        help='directory to save current model to')
    parser.add_argument('--data_dir', type=str, default='data/7scenes/chess',
                        help='directory to load the dataset')
    parser.add_argument('--reconstruct_accuracy', type=float, default=0.,
                        help='how accurate should the reconstruction be, float, negative and positive, '
                        'the smaller the more accurate')
    parser.add_argument('--dim_input', nargs=3, default=[96, 96, 3],
                        help='the height, width, and number of channel for input image, separated by space')
    parser.add_argument('--dim_reconstruct', nargs=3, default=[64, 64, 3],
                        help='the height, width, and number of channel for reconstructed image, separated by space')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    # # TODO reload&train is not implemented yet, as it was never really needed
    # parser.add_argument('--load_model', type=str, default=None,
    #                     help='Reload a model checkpoint and restore training.')
    args = parser.parse_args()

    # convert the list inputs into require format
    args.dim_input = tuple(map(int, args.dim_input))
    args.dim_reconstruct = tuple(map(int, args.dim_reconstruct))

    train(args)


def train(args):
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    json_path = os.path.join(args.model_dir, "model_specs.json")

    # get the dataset / simulated system
    dataloader = DataLoader(args.data_dir, img_width=args.dim_input[1], img_height=args.dim_input[0])

    # save the training flags
    args_dict = vars(args)
    args_dict['scaling_factor'] = [dataloader.norm_factor_xyz, dataloader.norm_factor_q]
    with open(json_path, 'w') as outfile:
        json.dump(args_dict, outfile, indent=4)
    
    # setup model
    model = Model(dim_observation=args.dim_observation,
                  batch_size=args.batch_size,
                  reconstruction_accuracy=args.reconstruct_accuracy,
                  image_size=args.dim_input,
                  reconstruct_size=args.dim_reconstruct,
                  training=True,
                  learning_rate=args.learning_rate)

    # train model
    model.train(dataloader,
                model_dir=args.model_dir,
                epoch=args.num_epochs,
                save_every=args.save_every)
    model.close()


if __name__ == "__main__":
    main()