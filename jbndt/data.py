
#TODO: load nlb data, merge held-in and held-out data to create one dataset.
#TODO: Write dataloader classes 
# %%
from nlb_tools.make_tensors import (make_train_input_tensors,
                                    make_eval_input_tensors,
                                    make_eval_target_tensors,
                                    save_to_h5)
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.evaluation import evaluate
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import h5py

logging.basicConfig(level=logging.INFO)

# %%
dataset_name = 'mc_maze'
datapath = Path('../datasets/000128/sub-Jenkins').absolute()
dataset = NWBDataset(datapath, skip_fields=['joint_ang', 'joint_vel', 'muscle_len'])
# resample to 5ms
dataset.resample(5)

train_dict = make_train_input_tensors(dataset,
                                    dataset_name=dataset_name, 
                                    trial_split='train', 
                                    include_behavior=True,
                                    save_file=False
                                    )
test_dict = make_train_input_tensors(dataset,
                                    dataset_name=dataset_name, 
                                    trial_split='val', 
                                    include_behavior=True,
                                    save_file=False
                                    )
# %%
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
train_behavior = train_dict['train_behavior'] # x and y vel

test_spikes_heldin = test_dict['train_spikes_heldin']
test_spikes_heldout = test_dict['train_spikes_heldout']
test_behavior = test_dict['train_behavior'] # x and y vel

