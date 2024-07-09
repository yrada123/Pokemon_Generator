from os.path import join, realpath, dirname
import yaml


def load_yaml(file_name):
    yaml_dir = dirname(realpath(__file__))
    with open(join(yaml_dir, file_name), 'r') as f:
        return yaml.safe_load(f)
