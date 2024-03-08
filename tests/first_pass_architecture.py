# %%
from ..jbndt.model import NDT

from ruamel.yaml import YAML
config_path = 'config.yaml'


yaml = YAML(typ='safe')
with open(config_path, 'r') as f:
    config = yaml.load(f)

model = NDT(config)
loss, rates = model([0])
# %%
