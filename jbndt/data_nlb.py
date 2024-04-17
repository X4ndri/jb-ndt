
#TODO: load nlb data, merge held-in and held-out data to create one dataset.
#TODO: Write dataloader classes 
# %%
from nlb_tools.make_tensors import (make_train_input_tensors,
                                    make_eval_input_tensors,
                                    make_eval_target_tensors,
                                    save_to_h5)
from torch.utils.data import TensorDataset, DataLoader
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.evaluation import evaluate
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import torch
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
                                    save_file=True
                                    )
test_dict = make_train_input_tensors(dataset,
                                    dataset_name=dataset_name, 
                                    trial_split='val', 
                                    include_behavior=True,
                                    save_file=True
                                    )
# %%
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
train_behavior = train_dict['train_behavior'] # x and y vel

test_spikes_heldin = test_dict['train_spikes_heldin']
test_spikes_heldout = test_dict['train_spikes_heldout']
test_behavior = test_dict['train_behavior'] # x and y vel

# Scenario 1: merge held-in and held-out data to create one dataset.
# %%
# merge held-in and held-out data to create one dataset.
# neuron 137:end are held-out neurons
train_spikes = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=2)
test_spikes = np.concatenate([test_spikes_heldin, test_spikes_heldout], axis=2)
# train behavior remains the same

# create data loader
X_train_tensor = torch.tensor(train_spikes, dtype=torch.float32)
y_train_tensor = torch.tensor(train_behavior, dtype=torch.float32)
X_test_tensor = torch.tensor(test_spikes, dtype=torch.float32)
y_test_tensor = torch.tensor(test_behavior, dtype=torch.float32)

# datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
