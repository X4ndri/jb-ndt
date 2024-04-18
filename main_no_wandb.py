""" Train without wandb. Use to test syntax and functionality of the code. Do not use for actual training.
author: aq
"""

#%%
import wandb
import sys
import os
from ruamel.yaml import YAML
from tests.architecture import *
from jbndt.data import *
from jbndt.cosmoothing_model import NDT
import torch
from torchsummary import summary
from sklearn.metrics import r2_score

config_path = '/home/aabdalq/projects/deeplearning/jbNDT/config.yaml'
yaml = YAML(typ='safe')
with open(config_path, 'r') as f:
    config = yaml.load(f)


device = torch.device('cuda')

train_data = train_data_module(device=device)
test_data = test_data_module(device=device)
ndt = NDT(config)



optimizer = torch.optim.AdamW(ndt.parameters(),
                              lr = config['lr_init'],
                              weight_decay = config['weight_decay'])

ndt.to(device)
for epoch in range(config['epochs']):
    ndt.train()
    for i, (X, y) in enumerate(train_data.dataloader):
        loss, nloss, bloss, logrates, velocities = ndt(X, [X, y], held_out_channels_count=50)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_batch_loss = loss.detach().cpu().numpy()
        
        #behavior
        inferred = velocities.detach().cpu().numpy()
        true = y.detach().cpu().numpy()
        train_r2 = r2_score(true.reshape(-1,2), inferred.reshape(-1,2), multioutput='uniform_average',)
    
    ndt.eval()
    for i, (X, y) in enumerate(test_data.dataloader):
        loss, nloss, bloss, logrates, velocities = ndt(X, [X, y])
        test_batch_loss = loss.detach().cpu().numpy()


    if epoch % 10 == 0:
        print(f'Epoch {epoch} train loss: {train_batch_loss} test loss: {test_batch_loss}')


# %%
