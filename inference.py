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
# model_path = "/home/aabdalq/projects/deeplearning/jbNDT/checkpoints/240417_164959_ndt_model_checkpoint.pth"
model_path = "/home/aabdalq/projects/deeplearning/jbNDT/checkpoints/epoch_740_240426_184632_ndt_model_checkpoint.pth"
h5_path = os.path.abspath('datasets/val_input_with_condition.h5')
requested_num_conditions = 10
requested_num_trials = 5

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
print('enforcing manual conditions')
selected_conditions = [21.0, 91.0, 30.0, 98.0,  7.0, 85.0, 73.0, 96.0, 95.0, 78.0]
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
        inferred_rates, inferred_velocity, latents = model(neural)
        inferred_trials[cond].append([inferred_rates.detach().numpy().squeeze(), inferred_velocity.detach().numpy().squeeze(), latents.detach().numpy().squeeze().squeeze()])

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
            axs[0].plot(position[:,0], position[:,1], color=colors[i], label=str(cond))
            continue
        axs[0].plot(position[:,0], position[:,1], color=colors[i])

for i, cond in enumerate(selected_conditions):
    for _, v, _ in inferred_trials[cond]:
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
        # smooth 
        x = np.array([smooth_gaussian(t[0][:,i], 20, 5) for i in range(t[0].shape[1])]).T
        n[i] = x

    avg = np.mean(n, axis=0)
    sem = np.std(n, axis=0) / np.sqrt(n.shape[0])
    true_psth[cond] = (avg, sem)


# %%
    
# for this, we need all trials per selected condition

# remember, condition_data has all trials [neural, velocity] per condition
# for each condition, pass all trials through the model, save the result in a dictionary
selected_conditions_inferred_trials = {}
all_latents = []
with torch.no_grad():
    for cond in selected_conditions:
        selected_conditions_inferred_trials[cond] = []
        n = np.zeros([len(condition_data[cond]), *condition_data[cond][0][0].shape])
        v = np.zeros([len(condition_data[cond]), *condition_data[cond][0][1].shape])
        l = np.zeros([len(condition_data[cond]), *[180, 64]])
        for i, trial in enumerate(condition_data[cond]):
            n_, v_ = trial
            neural = torch.Tensor(n_).unsqueeze(0)
            velocity = torch.Tensor(v_).unsqueeze(0)
            inferred_rates, inferred_velocity, latents = model(neural)
            inferred_rates = inferred_rates.detach().numpy().squeeze()
            smoothed_inferred_rates = np.array([smooth_gaussian(inferred_rates[:,i], 20, 5) for i in range(inferred_rates.shape[1])]).T
            n[i] = smoothed_inferred_rates
            v[i] = inferred_velocity.detach().numpy().squeeze()
            l[i] = latents.detach().numpy().squeeze().squeeze()
        
        all_latents.append(l)
        selected_conditions_inferred_trials[cond] = [n, v, l]

# %%
inferred_psth = {}
inferred_velocity = {}
inferred_latents = {}
for cond in selected_conditions:
    n, v, _ = selected_conditions_inferred_trials[cond]
    n_avg = np.mean(n, axis=0)
    n_sem = np.std(n, axis=0) / np.sqrt(n.shape[0])

    v_avg = np.mean(v, axis=0)
    v_sem = np.std(v, axis=0) / np.sqrt(v.shape[0])

    inferred_psth[cond] = (n_avg, n_sem)
    inferred_velocity[cond] = (v_avg, v_sem)


# %%
# plot PSTHs

# neuron 48, 63, 72
neuron = 77
f, axs = plt.subplots(requested_num_conditions, 1, figsize=(5, 10), sharey=True, sharex=True)
for i, cond in enumerate(selected_conditions):
    avg, sem = true_psth[cond]
    axs[i].plot(avg[:,neuron], color='black', label='True')
    axs[i].fill_between(np.arange(avg.shape[0]), avg[:,neuron] - sem[:,neuron], avg[:,neuron] + sem[:,neuron], color='black', alpha=0.5)
    
    avg, sem = inferred_psth[cond]
    axs[i].plot(avg[:,neuron], color=colors[i], label='Inferred')
    axs[i].fill_between(np.arange(avg.shape[0]), avg[:,neuron] - sem[:,neuron], avg[:,neuron] + sem[:,neuron], color=colors[i], alpha=0.5)


# %%
#PCA STUFF HERE
from sklearn.decomposition import PCA

n_components = 10
pca = PCA(n_components = n_components)
x = np.vstack(all_latents)
x_2d = x.reshape([-1, 64])
pca.fit(x_2d)
print(f"EXPLAINED VARIANCE: {pca.explained_variance_ratio_}")
latents_pca = {}
for cond in selected_conditions:
    latents = selected_conditions_inferred_trials[cond][2]
    latents = latents.reshape([-1, 64])
    condition_transformed_latents = np.array(pca.transform(latents)).reshape([-1, 180, n_components])
    latents_pca[cond] = condition_transformed_latents
latents_pca_avg = {}
for cond in selected_conditions:
    avg_latents = np.mean(latents_pca[cond], axis=0)
    latents_pca_avg[cond] = avg_latents

for i, cond in enumerate(selected_conditions):
    l_ = latents_pca_avg[cond]
    plt.plot(l_[:,2], l_[:,4], color = colors[i])
# %%
