"""
author: aq
The difference between cosmoothing_model.py and model.py, is that in model.py, we do not consider 
held-in held-out functionality, and concatenate all channels as input. In cosmoothing_model.py, we
consider held-in held-out functionality in the forward functions.
"""
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torch.optim as optim
import numpy as np

# %%

# -------------------------------------------------------------- #
#                        BUILDING BLOCKS
# -------------------------------------------------------------- #

                # ***********************
                # 1. Feed-forward Layer
                # ***********************

class FFL(nn.Module):
    def __init__(self, in_dim,
                hidden_dim,
                do_prob=0.2):
    
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(do_prob)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out

                # *****************************
                # 2. Multi-head Attention Layer
                # *****************************

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim,
                head_dim,
                num_heads,
                do_prob):
    
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gen_head_dim = gen_head_dim = head_dim * num_heads
        self.head_scaling = head_dim ** -0.5
        assert in_dim % head_dim == 0, f'in_dim must be divisible by gen_head_dim - orphan heads are {in_dim%head_dim}'
        
        self.ln = nn.LayerNorm(in_dim)
        self.fc_q = nn.Linear(in_dim, gen_head_dim, bias=False)
        self.fc_k = nn.Linear(in_dim, gen_head_dim, bias=False)
        self.fc_v = nn.Linear(in_dim, gen_head_dim, bias=False)
        self.fc_o = nn.Linear(gen_head_dim, in_dim)
        self.dropout = nn.Dropout(do_prob)
        self.relu = nn.ReLU()


    def attention(self, keys, queries, values, attn_mask=None):
        # transpose keys from B x num_heads x T x head_dim to B x num_heads x head_dim x T
        keys = torch.transpose(keys, -1, -2)


        # now get scores: queries have shape 
        # QUERIES: B x num_heads x T x head_dim
        # KEYS:    B x num_heads x head_dim x T
        # then -->
        #      -->   attn_scores: B x num_heads x T x T

        attn_scores = torch.matmul(queries, keys) * self.head_scaling
        # attention mask must then have shape: B x 1 x T x T

        if attn_mask is not None:
            attn_scores += attn_mask
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        # values have shape B x num_heads x T x head_dim
        # attn_scores have shape B x num_heads x T x T
        # make sure that the last two dimensions are the same
        attn_values = torch.matmul(attn_scores, values)
        # attn_values have shape B x num_heads x T x head_dim
        # transpose back to B x T x num_heads x head_dim
        attn_values = torch.transpose(attn_values, 1, 2).contiguous()
        # finally, reshape to B x T x gen_head_dim
        attn_values = attn_values.view(attn_values.size(0), -1, self.gen_head_dim)

        return attn_values
    

    def forward(self, x, attn_mask=None):
        batch_size, time, in_dim = x.size()
        x = self.ln(x)
        # x has dimension B x T x N, so after the linear transformation, it will be B x T x gen_head_dim
        # get queries, keys, values
        queries = self.fc_q(x)
        keys = self.fc_k(x)
        values = self.fc_v(x)
        # split heads so that they have dimension B x num_heads x T x head_dim
        transform = lambda x: x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # use map, much faster than a for loop
        queries, keys, values = map(transform, (queries, keys, values))
        # get attention scores: B x num_heads x T x T
        weighted_attention = self.attention(keys, queries, values, attn_mask)
        # weighted attention has shape B x T x gen_head_dim
        weighted_attention = self.fc_o(weighted_attention)
        # return weighted_attention
        return weighted_attention
    


# %%
# -------------------------------------------------------------- #
#                        Composites
# -------------------------------------------------------------- #
    
                # ***********************
                # 1. Encoder Layer
                # ***********************
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = FFL(
            in_dim=config['factor_dim'],
            hidden_dim=config['ffn_dim'],
            do_prob=config['ffn_dropout']
        )

        self.mha = MultiHeadAttention(
            in_dim=config['factor_dim'],
            head_dim=config['head_dim'],
            num_heads=config['num_heads'],
            do_prob=config['mha_dropout']
        )

        # now implement the dropouts after the attention and the ffn
        self.mha_dropout = nn.Dropout(config['mha_dropout'])
        self.ffn_dropout = nn.Dropout(config['ffn_dropout'])

    

    def forward(self, x, attn_mask=None):

        residual1 = torch.clone(x)
        out = self.mha(x, attn_mask=attn_mask)
        out = self.mha_dropout(out)
        out += residual1

        residual2 = torch.clone(out)
        out = self.ffn(out)
        out = self.ffn_dropout(out)
        out += residual2

        return out
    


                # ***********************
                # 2. NDT Model
                # ***********************
class NDT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forward_context = config['forward_context']
        self.backward_context = config['backward_context']
        # we can access those parameters from the config, but ...
        self.factor_dim = config['factor_dim']
        self.num_layers = config['num_layers']  
        self.head_dim = config['head_dim']
        self.num_heads = config['num_heads']
        # readin layer
        self.readin = nn.Linear(config['input_dim'], config['factor_dim'])
        # readout layerS
        self.neural_readout = nn.Linear(config['factor_dim'], config['input_dim'])
        self.behavioral_readout = nn.Linear(config['factor_dim'], config['behavioral_dim'])
        # Feed-forward layer
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_layers)])
        # layernorm
        self.ln = nn.LayerNorm(config['factor_dim'])
        # dropouts
        self.post_readin_dropout = nn.Dropout(config['post_readin_dropout'])
        self.post_positional_embedding_dropout = nn.Dropout(config['post_positional_embedding_dropout'])
        self.post_encoder_dropout = nn.Dropout(config['post_encoder_dropout'])
        # positional embeddings
        self.scale = torch.sqrt(torch.FloatTensor([config['factor_dim']])) # will scale the input by this to balance out positional embedding
        self.positional_embeddings = nn.Parameter(torch.zeros(config['sequence_length'], config['factor_dim']))
        self.positional_embeddings.requires_grad = True
        nn.init.xavier_uniform_(self.positional_embeddings)
        # define loss
        self.neural_criterion = nn.PoissonNLLLoss(reduction='none')
        self.behavioral_criterion = nn.MSELoss()
        self.loss_balance_factor = 1/config['loss_balance_factor']
        # ???????initialize weights and biases: not sure about ranges???????????
        self.neural_readout.bias.data.zero_()
        self.behavioral_readout.bias.data.zero_()
        self.neural_readout.weight.data.uniform_(-0.1, 0.1)
        self.behavioral_readout.weight.data.uniform_(-0.1, 0.1)
        self.readin.weight.data.uniform_(-0.1, 0.1) 

        # cache
        self.cache = None
        self.attn_mask = None

    def forward(self, x, y=None, held_out_channels_count=None):
        """
        x: input data
        y: labels, list of two tensors, first tensor is neural data, second tensor is behavioral data
        held_out_channels_count: number of channels to hold out starting from the last channel
        """


        # for now, y would contain neural labels and behavioral labels, y[0] is neural data, y[1] is behavioral data

        # modify the input and labels if necessary
        x, y = self.modify_batch(x, y, held_out_channels_count=held_out_channels_count)

        attn_mask = self.create_attention_mask(x)
        x = self.readin(x)
        x = self.post_readin_dropout(x)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        # scale x by the square root of the factor_dim
        x = x * self.scale.to(x.device)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        # add positional embeddings
        x = x + self.positional_embeddings
        x = self.post_positional_embedding_dropout(x)
        # encoder
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        x = self.ln(x)
        x = self.post_encoder_dropout(x)
        # readout
        logrates = self.neural_readout(x)
        velocities = self.behavioral_readout(x)
        if y is None:
            return torch.exp(logrates), velocities, x
        # compute loss
        neural_loss = self.neural_criterion(logrates, y[0])
        behavioral_loss = self.behavioral_criterion(velocities, y[1])

        loss = neural_loss.mean() + self.loss_balance_factor*behavioral_loss.mean()

        return loss, neural_loss.mean(), behavioral_loss.mean(), torch.exp(logrates), velocities


    def create_attention_mask(self, data):
        #TODO: implement attention mask with forward and backward context.
        # use cached version if already created.
        if self.attn_mask != None:
            return self.attn_mask

        ones = torch.ones(data.size(1), data.size(1), dtype=torch.float32, device=data.device)
        forw_mask = (torch.triu(ones, diagonal=-self.forward_context) == 1).transpose(0, 1)
        back_mask = (torch.triu(ones, diagonal=-self.backward_context) == 1)
        mask = (forw_mask & back_mask).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0)
        self.attn_mask = mask # cache the mask

        return mask


    def modify_batch(self, x, y, held_out_channels_count=None):
        #TODO: add held-in held-out functionality?
        #TODO: add different masking strategies?


        if held_out_channels_count is not None:
            # zero-out the heldout spikes
            x[:, -held_out_channels_count:] = 0
            y[1][:, -held_out_channels_count:] = 0

        return x, y
