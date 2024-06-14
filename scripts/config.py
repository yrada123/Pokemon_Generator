from os.path import join
import os.path as path
import yaml

# primary directories
PROJECT_DIR = path.dirname(path.dirname(path.realpath(__file__)))
DATA_DIR = join(PROJECT_DIR,'data')
CONFIGS_DIR = join(PROJECT_DIR,'configs')
NOTEBOOKS_DIR = join(PROJECT_DIR,'notebooks')
LOGS_DIR = join(PROJECT_DIR,'logs')

# secondary directories
ORIG_DATASET_DIR = 'orig_dataset'

# yaml handling
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

AE_YAML = load_yaml(path.join(CONFIGS_DIR, 'AE.yaml'))