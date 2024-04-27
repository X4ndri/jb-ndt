#%%
from sklearn.metrics import r2_score
from torchsummary import summary
from jbndt.callbacks import *
from ruamel.yaml import YAML
from jbndt.cosmoothing_model import NDT
from jbndt.data import *
from datetime import datetime
import wandb
import torch

SAVE_EVERY_N_EPOCHS = 500
GPU = 0
HELDOUT_CHANNEL_SIZE = 40
torch.cuda.set_device(GPU)
 
config_path = '../single_model.yaml'
device = torch.device('cuda')
train_data = train_data_module(device=device)
test_data = test_data_module(device=device)




# helper function
def save_model(model,
               optimizer,
               epoch,
               config,
               filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You may also want to save other relevant information like config
        'config': config
    }, filename
    )



# callbacks here
early_stopping = EarlyStoppingAccuracy(patience=40)

def train():
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%y%m%d_%H%M%S")  
    run = wandb.init()
    config=run.config

    ndt = NDT(config)
    ndt.to(device)
    
    optimizer = torch.optim.AdamW(ndt.parameters(),
                              lr = config['lr_init'],
                              weight_decay = config['weight_decay'])


    try:

        for epoch in range(config['epochs']):
            ndt.train()
            for i, (X, y) in enumerate(train_data.dataloader):
                loss, nloss, bloss, logrates, velocities = ndt(X, [X, y], held_out_channels_count=HELDOUT_CHANNEL_SIZE)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_batch_loss = loss.detach().cpu().numpy()
                
                #behavior
                inferred = velocities.detach().cpu().numpy()
                true = y.detach().cpu().numpy()
                train_r2 = r2_score(true.reshape(-1,2), inferred.reshape(-1,2), multioutput='uniform_average',)

                wandb.log({'train_batch_loss': train_batch_loss,
                        'train_neural_loss': nloss.detach().cpu().numpy(),
                        'train_behavioral_r2': train_r2,
                        'train_behavioral_loss': bloss.detach().cpu().numpy(),
                        'epoch': epoch})
            
            ndt.eval()
            test_losses = {'batch': [], 'neural': [], 'behavioral': [], 'r2': []}
            for i, (X, y) in enumerate(test_data.dataloader):
                loss, nloss, bloss, logrates, velocities = ndt(X, [X, y])
                test_batch_loss = loss.detach().cpu().numpy()

                #behavior
                inferred = velocities.detach().cpu().numpy()
                true = y.detach().cpu().numpy()
                test_r2 = r2_score(true.reshape(-1,2), inferred.reshape(-1,2), multioutput='uniform_average')
                test_losses['batch'].append(test_batch_loss)
                test_losses['neural'].append(nloss.detach().cpu().numpy())
                test_losses['behavioral'].append(bloss.detach().cpu().numpy())
                test_losses['r2'].append(test_r2)

                wandb.log({'test_batch_loss': test_batch_loss,
                        'test_neural_loss': nloss.detach().cpu().numpy(),
                        'test_behavioral_r2': test_r2,
                        'test_behavioral_loss': bloss.detach().cpu().numpy(),
                        'epoch': epoch})
            

            if epoch % 10 == 0:
                print(f'Epoch {epoch} train loss: {train_batch_loss} test loss: {test_batch_loss}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ndt.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # You may also want to save other relevant information like config
                    'config': dict(config)
                }, f'../checkpoints/epoch_{epoch}_{timestamp_str}_ndt_model_checkpoint.pth')


    except KeyboardInterrupt:
        print("INTERRUPTING")
        torch.save({
        'epoch': epoch,
        'model_state_dict': ndt.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You may also want to save other relevant information like config
        'config': dict(config)
    }, f'../checkpoints/{timestamp_str}_ndt_model_checkpoint.pth')


    # whenever epochs are done, save
    torch.save({
        'epoch': epoch,
        'model_state_dict': ndt.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You may also want to save other relevant information like config
        'config': dict(config)
    }, f'../checkpoints/{timestamp_str}_ndt_model_checkpoint.pth')



def main():

    # read sweep params
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        sweep_config = yaml.load(f)
    # creates a sweep id
    sweep_id = wandb.sweep(sweep_config, project="jbndt_cosmoothing_anna", entity='jbndt')
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()
# %%
