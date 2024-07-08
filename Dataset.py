from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import scripts.config as conf
import os
import shutil
import logging
import datetime
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pytorch_lightning as pl

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


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data_dir = root
        self.img_list = [x for x in os.listdir(root) if x.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img = Image.open(os.path.join(self.data_dir, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

class DataModule(pl.LightningDataModule):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.yaml = conf.VAE_YAML
        self.data_dir = self.yaml['data_params']['dir']
        self.batch_size = self.yaml['data_params']['batch_size']

    def setup(self, stage: str) -> None:
        size = self.yaml['data_params']['img_size']
        transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageDataset(root=self.data_dir, transform=transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)

    def prepare_data(self) -> None:
        '''Creating directory for original data and copy the data there.
           The directory is by default under 'data' directory.

        Args:
            output_dir (str): The name of the original dataset directory.
            debug (bool): Control logging elaboration.
        '''

        # Setting debugger and essential parameters
        if self.debug:
            logger.setLevel(logging.DEBUG)
        change_logger_format(logger, elaborated_format)
        logger.info("Preparing output directory (" + self.data_dir + ")")
        data_dir = conf.DATA_DIR
        output_dir = os.path.join(data_dir, self.data_dir)

        # Preparing output directory and handles the case when it already exists
        if os.path.basename(output_dir) in os.listdir(data_dir):
            logger.info('--- ' + output_dir + " already exists, skipping \'prepare data\' ---")
            return
        os.mkdir(output_dir)

        # Copying relevant data to output_dir
        logger.info("--- Starting copying process ---")
        for (root, _, files) in os.walk(os.path.join(data_dir, 'Pokemon Images DB')):
            for file in files:
                if 'new' not in os.path.basename(file):
                    shutil.copy(os.path.join(root, file), output_dir)
                    logger.debug("copied " + file)
        logger.info('--- ' + output_dir + " was set successfully ---")
