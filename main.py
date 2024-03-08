import sys
import os
from ruamel.yaml import YAML
from tests.architecture import *


config_path = './config.yaml'
yaml = YAML(typ='safe')
with open(config_path, 'r') as f:
    config = yaml.load(f)


first_pass(config)