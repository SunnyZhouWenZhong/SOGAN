from easydict import EasyDict as edict
import json
import os

config = edict()
config.TRAIN = edict()

config.TRAIN.batch_size = 8
config.TRAIN.early_stopping_num = 10
config.TRAIN.lr = 0.0001
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 10
config.TRAIN.n_epoch = 9999
config.TRAIN.sample_size = 50
config.TRAIN.g_alpha = 200  # weight for pixel loss
config.TRAIN.g_gamma = 100  # weight for gradient loss
config.TRAIN.g_adv = 1  # weight for frequency loss

config.TRAIN.seed = 100
config.TRAIN.epsilon = 0.000001

config.TRAIN.training_data_path = os.path.join('./data/Dataset', 'training.pickle')
config.TRAIN.val_data_path = os.path.join('./data/Dataset', 'validation.pickle')
config.TRAIN.testing_data_path = os.path.join('./data/Dataset', 'testing.pickle')
config.TRAIN.mask_Spiral_path = os.path.join('./data/mask/spiral/')
config.TRAIN.mask_Radial_path = os.path.join('./data/mask/radial/')
config.TRAIN.mask_Cartesian_path = os.path.join('./data/mask/cartesian/')
config.TRAIN.mask_Random_path = os.path.join('./data/mask/random/')

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")