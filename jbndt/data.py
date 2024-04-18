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
import os

logging.basicConfig(level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
class train_data_module:
    def __init__(self, 
                h5_path=os.path.abspath('../datasets/train_input_with_condition.h5'), 
                batch_size=32,
                device = torch.device('cuda')):
        
        self.h5_path = h5_path
        self.data = h5py.File(h5_path, 'r')
        self.conditions = self.data['conds'][:]
        
        recon = self.data['recon_data'][:]
        vel = self.data['vel'][:]

        self.X_train_tensor = torch.tensor(recon, dtype=torch.float32).to(device)
        self.y_train_tensor = torch.tensor(vel, dtype=torch.float32).to(device)

        self.dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)


class test_data_module:
    def __init__(self, 
                h5_path=os.path.abspath('../datasets/val_input_with_condition.h5'), 
                batch_size=32,
                device = torch.device('cuda')):
        
        self.h5_path = h5_path
        self.data = h5py.File(h5_path, 'r')
        self.conditions = self.data['conds'][:]
        
        recon = self.data['recon_data'][:]
        vel = self.data['vel'][:]

        self.X_train_tensor = torch.tensor(recon, dtype=torch.float32).to(device)
        self.y_train_tensor = torch.tensor(vel, dtype=torch.float32).to(device)

        self.dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

