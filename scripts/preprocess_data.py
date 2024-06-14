import scripts.config as conf
import os
import shutil
import logging
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# logger setting
log_filename = __name__.split('.')[-1] + ' ' + datetime.datetime.now().isoformat().replace(':','-') + '.log'
log_file = os.path.join(conf.LOGS_DIR, log_filename)
basic_format = "%(message)s"
elaborated_format = "%(asctime)s|%(levelname)s: " + basic_format
logging.basicConfig(filename=log_file, level=logging.INFO, format=elaborated_format)
logger = logging.getLogger(__name__)

def change_logger_format(logger, new_format):
    '''Change logger format

    Args:
        logger (Logger): an instance of logging library
        new_format (str): a format string
    '''
    for handler in logger.handlers:
        handler.setFormatter(new_format)

def copy_original_images_to_dir(output_dir=conf.ORIG_DATASET_DIR, debug=False):
    '''Creating directory for original data and copy the data there.
       The directory is by default under 'data' directory.

    Args:
        output_dir (str): The name of the original dataset directory.
        debug (bool): Control logging elaboration.
    '''

    # Setting debugger and essential parameters
    if debug:
        logger.setLevel(logging.DEBUG)
    change_logger_format(logger, elaborated_format)
    logger.info("Preparing output directory (" + output_dir + ")")
    data_dir = conf.DATA_DIR
    output_dir = os.path.join(data_dir, output_dir)

    # Preparing output directory and handles the case when it already exists
    if os.path.basename(output_dir) in os.listdir(data_dir):
        print(output_dir + "already exists, are you sure you want to override it? y/n")
        ans = input()
        cond = True if ans != 'y' and ans != 'n' else False
        while cond:
            ans = input()
            cond = True if ans != 'y' and ans != 'n' else False
        if ans == 'n':
            logger.warning(output_dir + ' was left unchanged')
            logger.setLevel(logging.INFO)
            return
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Copying relevant data to output_dir
    logger.info("--- Starting copying process ---")
    for (root,_,files) in os.walk(os.path.join(data_dir,'Pokemon Images DB')):
        for file in files:
            if 'new' not in os.path.basename(file):
                shutil.copy(os.path.join(root,file), output_dir)
                logger.debug("copied " + file)
    logger.info('--- ' +  output_dir + " was set successfully ---")

def load_dataset() -> DataLoader:
    ae_yaml = conf.AE_YAML

    transform = transforms.Compose([transforms.ToTensor,
                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(root=ae_yaml['data_params']['dir'], transform=transform)
    return DataLoader(dataset=dataset, batch_size=ae_yaml['data_params']['batch_size'])