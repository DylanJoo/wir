import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class crossattentionlayer(nn.Module):
    def __init__(self, config, zero_init=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention # no V and no O
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if zero_init:
            self.q_proj.weight.data.zero_()
            self.q_proj.bias.data.zero_()
            self.k_proj.weight.data.zero_()
            self.k_proj.bias.data.zero_()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        beta = 0.0,
        **kwargs
    ) -> torch.Tensor:

        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        
        def shape(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        q = shape(self.q_proj(hidden_states))
        k = shape(self.k_proj(encoder_hidden_states))
        v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )

        # select only k
        attention_probs = F.softmax(attention_scores, dim=-1)
        # attention_probs = F.gumbel_softmax(attention_scores, dim=-1, hard=True)
        
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)

        context_hidden_states = context_layer + hidden_states
        
        return (context_hidden_states, attention_scores, ) 
