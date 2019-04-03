
## Introduction
This is the code base for our work of Generative Map at <https://arxiv.org/abs/1902.11124>.  
It is an effort to combine generative model (in particular, [Variational Auto-Encoders](https://arxiv.org/abs/1312.6114)) and the classic Kalman filter for generation with localization.  
For more details, please refer to [our paper on arXiv](https://arxiv.org/abs/1902.11124).

## Dependencies
1. Python3, and the following packages  
```
pip install numpy scikit-image tqdm tensorflow==1.13.1
```

2. Datasets:  
  * 7-Scenes: Download from [their website](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).  
  * RobotCar: Register & Download from [here](https://robotcar-dataset.robots.ox.ac.uk/).  

## Usage
For training, one example use can be found below. The `PATH_TO_DATA_DIR` points to the data folder, for example in 7-Scenes it is the folder where you can find `TrainSplit.txt`, `TestSplit.txt`, and folders of `seq-xx`.  
```
python train.py --data_dir <PATH_TO_DATA_DIR>
```

For evaluation, one example of estimation with generation using a constant model is  
```
python eval.py --model_dir <PATH_TO_YOUR_MODEL_DIR_WITH_CHECKPOINTS> --generation_mode 2 --no_model
```

## Detail Usage
```
usage: train.py [-h] [--training_size TRAINING_SIZE]
                [--dim_observation DIM_OBSERVATION] [--batch_size BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--save_every SAVE_EVERY]
                [--model_dir MODEL_DIR] [--data_dir DATA_DIR]
                [--reconstruct_accuracy RECONSTRUCT_ACCURACY]
                [--dim_input DIM_INPUT DIM_INPUT DIM_INPUT]
                [--dim_reconstruct DIM_RECONSTRUCT DIM_RECONSTRUCT DIM_RECONSTRUCT]
                [--learning_rate LEARNING_RATE] [--use_robotcar]

optional arguments:
  -h, --help            show this help message and exit
  --training_size TRAINING_SIZE
                        size of the training set, sampled from the given
                        sequences
  --dim_observation DIM_OBSERVATION
                        dimension of the latent representation ("observation")
  --batch_size BATCH_SIZE
                        mini-batch size for training
  --num_epochs NUM_EPOCHS
                        number of epochs for training
  --save_every SAVE_EVERY
                        intermediate saving frequency by epochs
  --model_dir MODEL_DIR
                        directory to save current model to
  --data_dir DATA_DIR   directory to load the dataset
  --reconstruct_accuracy RECONSTRUCT_ACCURACY
                        how accurate should the reconstruction be, float,
                        negative and positive, the smaller the more accurate
  --dim_input DIM_INPUT DIM_INPUT DIM_INPUT
                        the height, width, and number of channel for input
                        image, separated by space
  --dim_reconstruct DIM_RECONSTRUCT DIM_RECONSTRUCT DIM_RECONSTRUCT
                        the height, width, and number of channel for
                        reconstructed image, separated by space
  --learning_rate LEARNING_RATE
                        learning rate
  --use_robotcar        if the robotcar data loader will be used, since it is
                        stored in a different format.
```
  
```
usage: eval.py [-h] [--model_dir MODEL_DIR] [--state_var STATE_VAR]
               [--visualize_dir VISUALIZE_DIR] [--video] [--no_model]
               [--generation_mode GENERATION_MODE] [--all] [--seq_idx SEQ_IDX]
               [--deviation DEVIATION]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        directory to load model from. If not provided, the
                        most recent dir in the default path will be used,
                        raise an Error if failed.
  --state_var STATE_VAR
                        estimated state transition variance for the model used
                        by extended Kalman filter, it is assumed to be
                        independent and stay the same for all dimension
  --visualize_dir VISUALIZE_DIR
                        directory to save the visualization output + video (if
                        apply)
  --video               if a video should be recorded comparing real and
                        estimated reconstructed images, together with
                        localization results
  --no_model            if the true state transition will be hidden to test
                        localization based only on images.
  --generation_mode GENERATION_MODE
                        0: equidistant sample image generation; 1: Kalman
                        filter estimation for localization; 2: perform both
  --all                 evaluate all the validation sequences and report the
                        performance w/o a picture.
  --seq_idx SEQ_IDX     evaluate the given indexed sequence in the text.
  --deviation DEVIATION
                        deviation across all state dimension for robust test.
```


