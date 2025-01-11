import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CrossAttentionLayer(nn.Module):
    def __init__(self, config, zero_init=False, mono_attend=False, output_layer=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention (adapted from BertSelfAttention)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Cross attention output layer (adapted from BertSelfOutput)
        self.output_layer = output_layer

        if zero_init:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.output_proj.bias)

        self.mono_attend = mono_attend
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        **kwargs
    ) -> torch.Tensor:

        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        
        def shape(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)

        # B N_head L H_head
        q = shape(self.q_proj(hidden_states))
        k = shape(self.k_proj(encoder_hidden_states))
        v = shape(self.v_proj(encoder_hidden_states))
        # k = shape(encoder_hidden_states)
        # v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )

        if self.mono_attend:
            attention_probs = F.gumbel_softmax(attention_scores, dim=-1, hard=True)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1) # B N_head Lq Lk
        
        context_layer = torch.matmul(attention_probs, v) # B N_head Lq Lk x B N_head Lk H_head = B N_head Lq H_head
        context_layer = context_layer.permute(0, 2, 1 ,3).contiguous() # B L_q N_head H_head
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size) # B L_q H

        # output layer
        if self.output_layer:
            attention_output = context_layer + hidden_states
        else:
            attention_output = context_layer
        
        return (attention_output, attention_scores)

class CrossAttentionSelector(nn.Module):
    def __init__(self, config, zero_init=False, mono_attend=False, output_layer=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Cross attention (adapted from BertSelfAttention)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Cross attention output layer (adapted from BertSelfOutput)
        self.output_layer = output_layer

        if zero_init:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.output_proj.bias)

        self.mono_attend = mono_attend
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        **kwargs
    ) -> torch.Tensor:

        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        
        def shape(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).permute(0, 2, 1, 3)

        # B N_head L H_head
        q = shape(self.q_proj(hidden_states))
        k = shape(self.k_proj(encoder_hidden_states))
        v = shape(self.v_proj(encoder_hidden_states))
        # k = shape(encoder_hidden_states)
        # v = shape(encoder_hidden_states)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            encoder_attention_mask[:, None, None, :] == 0,
            float('-inf')
        )

        if self.mono_attend:
            attention_probs = F.gumbel_softmax(attention_scores, dim=-1, hard=True)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1) # B N_head Lq Lk
        
        context_layer = torch.matmul(attention_probs, v) # B N_head Lq Lk x B N_head Lk H_head = B N_head Lq H_head
        context_layer = context_layer.permute(0, 2, 1 ,3).contiguous() # B L_q N_head H_head
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size) # B L_q H

        # output layer
        if self.output_layer:
            attention_output = context_layer + hidden_states
        else:
            attention_output = context_layer
        
        return (attention_output, attention_scores)
