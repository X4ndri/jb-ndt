from ruamel.yaml import YAML
from jbndt.model import NDT
import jbndt.data as data

# read in the config file
config_path = './config.yaml'
yaml = YAML(typ='safe')
with open(config_path, 'r') as f:
    config = yaml.load(f)


train_data = data.train_data_module(batch_size=config['batch_size']).dataloader
test_data = data.test_data_module(batch_size=config['batch_size']).dataloader

# dataloaders sanity check
for X, y in train_data:
    print(X.shape, y.shape)
    break

