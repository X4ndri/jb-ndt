# %%
from jbndt.cosmoothing_model import NDT
from scipy.signal import convolve
from scipy.signal import gaussian
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import h5py
import os

# %%
# model_path = "/home/aabdalq/projects/deeplearning/jbNDT/checkpoints/240417_164959_ndt_model_checkpoint.pth"
model_path ="/home/aabdalq/projects/deeplearning/jbNDT/checkpoints/240417_225216_ndt_model_checkpoint.pth" 
model_path = "/home/aabdalq/projects/deeplearning/jbNDT/checkpoints/240417_230852_ndt_model_checkpoint.pth"
h5_path = os.path.abspath('datasets/train_input_with_condition.h5')
requested_num_conditions = 10
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
selected_conditions = [21.0, 91.0, 30.0, 98.0,  7.0, 85.0, 73.0, 96.0, 95.0, 78.0]
# selected_conditions = np.concatenate([selected_conditions, np.random.choice(list(condition_data.keys()), 4, replace=False)])
print(selected_conditions)
for cond in condition:
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


# generate an array of unique colors, one for each condition
colors = plt.cm.turbo(np.linspace(0, 1, len(selected_conditions))) 
# plot trials, colored by condition
f, axs = plt.subplots(2, 1, figsize=(10, 10))
for ax in axs:
    #ax.set_aspect('equal')
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


handles = []  # Collect handles for legend
labels = []   # Collect labels for legend

for i, cond in enumerate(selected_conditions):
    for j, (n, v) in enumerate(true_trials[cond]):
        position = np.cumsum(v, axis=0)
        if j == requested_num_trials - 1:
            line, = axs[0].plot(position[:,0], position[:,1], color=colors[i])
            handles.append(line)
            labels.append(str(cond))
            continue
        axs[0].plot(position[:,0], position[:,1], color=colors[i])

for i, cond in enumerate(selected_conditions):
    for _, v in inferred_trials[cond]:
        position = np.cumsum(v, axis=0)
        axs[1].plot(position[:,0], position[:,1], color=colors[i], label=str(int(cond)))

# Add legends for each condition
axs[0].legend(handles, labels, loc='upper right')
axs[1].legend(handles, labels, loc='upper right')

#axs[1].legend(selected_conditions, loc='upper right')  # Assuming selected_conditions is the list of condition names

plt.tight_layout()
plt.show()







# %%
###############################################
#               Generate PSTHs
###############################################

# for each condition, average neural activity keeping track of the standard error of the mean
true_psth = {}
for cond in tqdm(condition[:50]):
    n = np.zeros([len(condition_data[cond]), *true_trials[cond][0][0].shape])
    for i, t in enumerate(condition_data[cond]):
        for j, x in enumerate(t[0].squeeze().T):
            n[i, :, j] = smooth_gaussian(x, 20, 3)
    avg = np.mean(n, axis=0)
    sem = np.std(n, axis=0) / np.sqrt(n.shape[0])
    true_psth[cond] = (avg.squeeze(), sem.squeeze())

# %%
    
# for this, we need all trials per selected condition

# remember, condition_data has all trials [neural, velocity] per condition
# for each condition, pass all trials through the mode, save the result in a dictionary
selected_conditions_inferred_trials = {}
with torch.no_grad():
    for cond in tqdm(condition[:50]):
        selected_conditions_inferred_trials[cond] = []
        for trial in condition_data[cond]:
            n, v = trial
            neural = torch.Tensor(n).unsqueeze(0)
            velocity = torch.Tensor(v).unsqueeze(0)
            inferred_rates, inferred_velocity = model(neural)
            selected_conditions_inferred_trials[cond].append([inferred_rates.detach().numpy(), inferred_velocity.detach().numpy().squeeze()])

# %%
inferred_psth = {}
for cond in tqdm(condition[:50]):
    n = np.zeros([len(selected_conditions_inferred_trials[cond]), *inferred_rates.squeeze().shape])
    for i, t in enumerate(selected_conditions_inferred_trials[cond]):
        for j, x in enumerate(t[0].squeeze().T):
            n[i, :, j] = smooth_gaussian(x, 20, 3)
    avg = np.mean(n, axis=0)
    sem = np.std(n, axis=0) / np.sqrt(n.shape[0])
    inferred_psth[cond] = (avg.squeeze(), sem.squeeze())
# %%
# neuron = 37
neuron = 170
f, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(20, 20))
cond_indices = [1]
cond = condition[cond_indices][0]
avgt, semt = true_psth[cond]
avgi, semi = inferred_psth[cond]

for i, ax in enumerate(axs.flatten()):
    #ax.set_aspect('equal')
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    i *= -1
    ax.plot(avgt[:, i])
    ax.fill_between(np.arange(avgt.shape[0]), avgt[:, i] - semt[:, i], avgt[:, i] + semt[:, i], alpha=0.5)
    ax.plot(avgi[:, i])
    ax.fill_between(np.arange(avgi.shape[0]), avgi[:, i] - semi[:, i], avgi[:, i] + semi[:, i], alpha=0.5)


#%%
cond_indices = [ 26, 3, 4]
conds_ = condition[cond_indices]
for i, cond in enumerate(conds_):
    avgi, semi = inferred_psth[cond]
    avgt, semt = true_psth[cond]
    axs[0].plot(avgt[:, neuron], color=colors[i])
    axs[0].fill_between(np.arange(avgt.shape[0]), avgt[:, neuron] - semt[:, neuron], avgt[:, neuron] + semt[:, neuron], color=colors[i], alpha=0.5)
    axs[1].plot(avgi[:, neuron], color=colors[i])
    axs[1].fill_between(np.arange(avgi.shape[0]), avgi[:, neuron] - semi[:, neuron], avgi[:, neuron] + semi[:, neuron], color=colors[i], alpha=0.5)

# %%
cond_indices = [1]
cond = condition[cond_indices][0]
modulated_neurons = [11, 16, 17]
modulated_neurons = -1* np.array(modulated_neurons)
f, axs = plt.subplots(3,1, sharex=True, sharey=True)
avgt, semt = true_psth[cond]
avgi, semi = inferred_psth[cond]
for i, ax in enumerate(axs.flatten()):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(avgt[:, modulated_neurons[i]], label='True', color=colors[0])
    ax.fill_between(np.arange(avgt.shape[0]), avgt[:, modulated_neurons[i]] - semt[:, modulated_neurons[i]], avgt[:, modulated_neurons[i]] + semt[:, modulated_neurons[i]], alpha=0.5, color=colors[0])
    ax.plot(avgi[:, modulated_neurons[i]], label='Inferred', color=colors[3])
    ax.fill_between(np.arange(avgi.shape[0]), avgi[:, modulated_neurons[i]] - semi[:, modulated_neurons[i]], avgi[:, modulated_neurons[i]] + semi[:, modulated_neurons[i]], alpha=0.5, color=colors[3])
axs[0].legend(loc='upper left')
plt.tight_layout()
plt.show()
