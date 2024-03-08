
import sys
import os
ptp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f'appending {ptp} to path...')
sys.path.append(ptp)
from jbndt.model import NDT
from torchsummary import summary
import torch


'''
write unit tests for the architecture of the model or for sanity checks.
'''

def first_pass(config):
    model = NDT(config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")
    model.to(device)
    print(summary(model, (config['sequence_length'], config['input_dim']), device='cuda' if device == torch.device('cuda') else 'cpu'))



def second_pass():
    pass


# TODO: add tests here