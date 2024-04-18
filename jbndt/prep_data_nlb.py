""" 
preps data and saves train and val data to disk as h5 files. RUN ONCE
"""

#TODO: Write dataloader classes --> done in data.py

# %%
from nlb_tools.make_tensors import (make_train_input_tensors,
                                    make_eval_input_tensors,
                                    make_eval_target_tensors,
                                    save_to_h5)
from torch.utils.data import TensorDataset, Dataset, DataLoader
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

def create_dataset_and_save_to_disk(dataset, h5_save_path):
    spk_field = "spikes"
    hospk_field="heldout_spikes"
    align_field = "move_onset_time"
    align_window= (-250, 450)
    align_field_fwd= "move_onset_time"
    align_window_fwd= (450,650)

    dataloaders = {}

    for trial_split in ["train", "val"]:
        split_to_mask = lambda x: (dataset.trial_info.split == x) if isinstance(x, str) else x
        trial_mask = split_to_mask(trial_split)
        allow_nans = trial_split != "train"
        trial_data = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field = align_field, align_range = align_window)
        trial_data = add_conds_to_trial_data(trial_data, dataset)

        trial_data_fwd = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field = align_field_fwd, align_range = align_window_fwd)
        trial_data_fwd = add_conds_to_trial_data(trial_data_fwd, dataset)

        grouped = list(trial_data.groupby('trial_id', sort=False))
        grouped_fwd = list(trial_data_fwd.groupby('trial_id', sort=False))

        heldin = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped]))
        heldout = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped]))
        heldin_fwd = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped_fwd]))
        heldout_fwd = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped_fwd]))

        vel= torch.Tensor(np.stack([trial["hand_vel"].to_numpy() for _, trial in grouped]))
        vel_fw = torch.Tensor(np.stack([trial["hand_vel"].to_numpy() for _, trial in grouped_fwd]))
        vel = torch.cat([vel, vel_fw], dim=1)
        conds = torch.Tensor(np.stack([trial["trial_cond"].to_numpy() for _, trial in grouped]))

        heldin_full = torch.cat([heldin, heldin_fwd], dim=1)
        heldout_full = torch.cat([heldout, heldout_fwd], dim = 1)
        recon_data = torch.cat([heldin_full, heldout_full], dim =2)

        # dump to h5
        filename = h5_save_path.joinpath(f"{trial_split}_input_with_condition.h5").as_posix()
        with h5py.File(filename, 'w') as f:
            f.create_dataset("heldin", data=heldin_full.numpy())
            f.create_dataset("heldout", data=heldout_full.numpy())
            f.create_dataset("recon_data", data=recon_data.numpy())
            f.create_dataset("vel", data=vel.numpy())
            f.create_dataset("conds", data=conds.numpy())



def add_conds_to_trial_data(trial_data_in, dataset_in):
    cond_fields = ['trial_type', 'trial_version']
    combinations = sorted(dataset_in.trial_info[cond_fields].dropna().set_index(cond_fields).index.unique().tolist())
    combinations = np.array(combinations)
    trial_data = trial_data_in.copy()
    trial_info = dataset_in.trial_info
    trial_nums = trial_info.trial_id.values
    trial_data['trial_cond'] = np.zeros(len(trial_data))
    for i,comb in enumerate(combinations):
        # Need a list of all the trial_ids that match cond
        flag1 = trial_info.trial_type.values == comb[0]
        flag2 = trial_info.trial_version.values == comb[1]
        flag3 = np.logical_and(flag1, flag2)
        trial_flag =np.where(flag3) # a list of indices in trial_info that
        cond_trials = trial_nums[trial_flag]
        trial_data.loc[np.isin(trial_data.trial_id, cond_trials),'trial_cond'] = i
    return trial_data


# %%
dataset_name = 'mc_maze'
datapath = Path('../datasets/000128/sub-Jenkins').absolute()
print(datapath)
dataset = NWBDataset(datapath, skip_fields=['joint_ang', 'joint_vel', 'muscle_len'])
dataset.resample(5)

# %%
dataset_directory = datapath.parent.parent
#%%
logging.info('creating...')
# resample to 5ms
create_dataset_and_save_to_disk(dataset, dataset_directory)

# if __name__ == "__main__":
#     datapath = Path('datasets/000128/sub-Jenkins').absolute()
#     print(datapath)
#     dataset = NWBDataset(datapath, skip_fields=['joint_ang', 'joint_vel', 'muscle_len'])
#     dataset.resample(5)
#     
#     
#     filename = Path(__file__)
#     dataset_directory = filename.parent.joinpath('datasets')
# 
#     logging.info('creating...')
# # resample to 5ms
#     create_dataset_and_save_to_disk(dataset, dataset_directory)


# %%
