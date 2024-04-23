# Setting up environment
1. Use the nlb environment as a first step -> we'll need nlb-tools later on. It's best to use python3.7.
2. Update environment by the requirements file in this repo: <br>
```pip install -r requirements.txt```
# Downloading data
1. cd datasets
2. ''' dandi download https://gui.dandiarchive.org/#/dandiset/000128 '''
# Directory map
1. The model architecture is in ```jbndt/model.py```
2. Data refinement and loaders go in ```jbndt/data.py```
3. `sweep.yaml` will be used later to run wandb HP searches
# MISC
1. There are a bunch of `#TODO:` tags throughout, we'll hunt these down one by one as we pass unittests.