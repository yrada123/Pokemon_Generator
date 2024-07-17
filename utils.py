import os
from os.path import join, basename, dirname
import shutil
import random


def split_train_val(data_dir_path='data/dataset', train_per=0.8):
    """ Split data to train and val directories

        Args:
            data_dir_path (string): Fullpath or relative path to the data directory (from root)
            train_per (float): Percentage of data for training (0,1)
        :return:
    """

    assert 0 <= train_per <= 1, "train_per must be in [0,1] range"
    data_dir_path = join(dirname(__file__), data_dir_path)

    # get file names for validation
    data = os.listdir(data_dir_path)
    random.shuffle(data)
    div_point = int(len(data) * train_per)
    val_data = data[div_point:]

    # add "train_" to the beginning of the directory name if "train" is not in the name
    path = dirname(data_dir_path)
    dir_name = basename(data_dir_path)

    if 'train' not in dir_name:
        dir_name = 'train_' + dir_name
        os.rename(data_dir_path, join(path, dir_name))

    # creates validation directory
    val_dir_path = join(path, dir_name.replace("train", "val"))
    os.mkdir(join(path, val_dir_path))

    # move validation files to the new directory
    for v in val_data:
        src = join(path, dir_name, v)
        shutil.move(src, val_dir_path)
