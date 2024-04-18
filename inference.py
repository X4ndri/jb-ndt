# %%
from jbndt.cosmoothing_model import NDT
from scipy.signal import convolve
from scipy.signal import gaussian
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import h5py
import os

# %%
model_path = "/home/aabdalq/projects/deeplearning/jbNDT/checkpoints/240417_164959_ndt_model_checkpoint.pth"
h5_path = os.path.abspath('datasets/train_input_with_condition.h5')
requested_num_conditions = 5
requested_num_trials = 10

# %%
# helper functions
def smooth_gaussian(data, window_size, sigma):
    gauss = gaussian(window_size, sigma)
    gauss /= np.sum(gauss)
    return convolve(data, gauss, mode='same')


# %%
checkpoint = torch.load(model_path)
model = NDT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# %%
# extract data
data = h5py.File(h5_path, 'r')
neural_data = data['recon_data'][:]
velocity = data['vel'][:]
condition = data['conds'][:][:,0]
data.close()
# %%
condition_data = {}
for i in range(len(condition)):
    cond = condition[i]
    if cond not in condition_data:
        condition_data[cond] = []
    condition_data[cond].append((neural_data[i], velocity[i]))

# %%
###############################################
#               Generate Trajectories
###############################################
# select conditions and trials per condition
true_trials = {}
selected_conditions = np.random.choice(list(condition_data.keys()), requested_num_conditions, replace=False)
print(selected_conditions)
for cond in selected_conditions:
    selected_trial_indices = np.random.choice(np.arange(len(condition_data[cond])), requested_num_trials, replace=False)
    true_trials[cond] = [condition_data[cond][i] for i in selected_trial_indices]

inferred_trials = {}
for cond in selected_conditions:
    inferred_trials[cond] = []
# pass trials through model
for cond in selected_conditions:
    for trial in true_trials[cond]:
        n, v = trial
        neural = torch.Tensor(n).unsqueeze(0)
        velocity = torch.Tensor(v).unsqueeze(0)
        inferred_rates, inferred_velocity = model(neural)
        inferred_trials[cond].append([inferred_rates.detach().numpy().squeeze(), inferred_velocity.detach().numpy().squeeze()])

# %%
# generate an array of unique colors, one for each condition
colors = plt.cm.turbo(np.linspace(0, 1, len(selected_conditions))) 
# plot trials, colored by condition
f, axs = plt.subplots(2, 1, figsize=(10, 10))
for ax in axs:
    ax.set_aspect('equal')
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

for i, cond in enumerate(selected_conditions):
    for j, (n, v) in enumerate(true_trials[cond]):
        # integrate velocity to get position
        position = np.cumsum(v, axis=0)
        if j == requested_num_trials - 2:
            print('yee')
            axs[0].plot(position[:,0], position[:,1], color=colors[i], label=str(cond))
            continue
        axs[0].plot(position[:,0], position[:,1], color=colors[i])

for i, cond in enumerate(selected_conditions):
    for _, v in inferred_trials[cond]:
        position = np.cumsum(v, axis=0)
        axs[1].plot(position[:,0], position[:,1], color=colors[i])

plt.legend()
plt.tight_layout












# %%
###############################################
#               Generate PSTHs
###############################################

# for each condition, average neural activity keeping track of the standard error of the mean
true_psth = {}
for cond in selected_conditions:
    n = np.zeros([requested_num_trials, *true_trials[cond][0][0].shape])
    for i, t in enumerate(true_trials[cond]):
        n[i] = t[0]
    avg = np.mean(n, axis=0)
    sem = np.std(n, axis=0) / np.sqrt(n.shape[0])
    true_psth[cond] = (avg, sem)


# %%
    
# for this, we need all trials per selected condition

# remember, condition_data has all trials [neural, velocity] per condition
# for each condition, pass all trials through the mode, save the result in a dictionary
selected_conditions_inferred_trials = {}
with torch.no_grad():
    for cond in selected_conditions:
        selected_conditions_inferred_trials[cond] = []
        for trial in condition_data[cond]:
            n, v = trial
            neural = torch.Tensor(n).unsqueeze(0)
            velocity = torch.Tensor(v).unsqueeze(0)
            inferred_rates, inferred_velocity = model(neural)
            selected_conditions_inferred_trials[cond].append([inferred_rates.detach().numpy(), inferred_velocity.detach().numpy().squeeze()])

# %%
inferred_psth = {}
for cond in selected_conditions:
    n = np.zeros([len(selected_conditions_inferred_trials[cond]), *inferred_trials[cond][0][0].shape])
    for i, t in enumerate(selected_conditions_inferred_trials[cond]):
        n[i] = np.array([smooth_gaussian(x, 20, 3) for x in t[0]])
    avg = np.mean(n, 10,3, axis=0)
    sem = np.std(n, 10,3, axis=0) / np.sqrt(n.shape[0])
    inferred_psth[cond] = (avg.squeeze(), sem.squeeze())
# %%
neuron = 15

# %%

f, axs = plt.subplots(2, 1)

for i, cond in enumerate(selected_conditions[:3]):
    avgi, semi = inferred_psth[cond]
    avgt, semt = true_psth[cond]
    axs[0].plot(smooth_gaussian(avgi[:,neuron], 20, 3), color=colors[i])
    axs[1].plot(smooth_gaussian(avgt[:,neuron], 20, 3), color=colors[i])

# %%
a = smooth_gaussian(ni[:,neuron], 10, 3)
# %%
