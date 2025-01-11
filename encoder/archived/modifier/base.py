import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
import math

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
# from .dist_utils import gather

@dataclass
class BiencoderOutput(BaseModelOutput):
    qembs: torch.FloatTensor = None
    dembs: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class FFModifierHead(nn.Module):
    def __init__(self, input_size, output_size=None, zero_init=False, **kwargs):
        super().__init__()
        output_size = (output_size or input_size)
        self.fc_1 = nn.Linear(input_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        if zero_init:
            self.fc_1.weight.data.zero_()
            self.fc_1.bias.data.zero_()
            self.fc_2.weight.data.zero_()
            self.fc_2.weight.data.zero_()
        dropout_prob = kwargs.pop("dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob != 0.0 else nn.Identity()

    def forward(self, qremb, qfemb):
        emb = torch.cat( (qremb, qfemb), -1)
        output = self.dropout(emb)
        if output.dtype != self.fc_1.weight.dtype:
            output = output.to(self.fc_1.weight.dtype)
        output = self.fc_2(self.fc_1(output))
        return output

class PlusModifierHead(nn.Module):
    def __init__(self, input_size, output_size=None, zero_init=False, **kwargs):
        super().__init__()
        output_size = (output_size or input_size)
        self.fc = None
        if zero_init:
            self.fc = nn.Linear(input_size, input_size)
            self.fc.weight.data.zero_()
            self.fc.bias.data.zero_()

    def forward(self, qremb, qfemb):
        if self.fc is None:
            return (qremb + qfemb)/2
        else:
            qfemb = self.fc(qfemb)
            return qremb + qfemb

class FeedbackQueryModifier(nn.Module):

    def __init__(
        self, 
        opt, 
        qr_encoder, 
        qf_encoder=None,
        d_encoder=None,
        n_candidates=None,
        fusion_type='ff',
        zero_init=False
    ):
        super().__init__()
        self.opt = opt
        self.is_ddp = dist.is_initialized()
        self.tau = opt.tau
        self.n_candidates = n_candidates

        self.qr_encoder = qr_encoder
        self.qf_encoder = (qf_encoder or qr_encoder)
        self.d_encoder = (d_encoder or qr_encoder)
        if fusion_type == 'ff':
            self.modifier = FFModifierHead(
                qr_encoder.config.hidden_size + qf_encoder.config.hidden_size,
                qr_encoder.config.hidden_size,
                zero_init=zero_init
            )
        if fusion_type == 'plus':
            self.modifier = PlusModifierHead(
                qf_encoder.config.hidden_size,
                qf_encoder.config.hidden_size,
                zero_init=zero_init
            )

        for n, p in self.named_parameters():
            if 'd_encoder' in n:
                p.requires_grad = False
            elif 'qr_encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(
        self, 
        q_tokens, 
        q_masks, 
        d_tokens=None, 
        d_masks=None, 
        **kwargs
    ):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('max_num_steps', n_segments)
        batch_size = q_tokens[0].size(0)

        # encode query request and query feedback
        qembs = []
        for i in range(max_num_steps):
            if i == 0:
                qemb = self.qr_encoder(q_tokens[0], q_masks[0]).emb
                qfemb = self.qf_encoder(q_tokens[0], q_masks[0]).emb  # B H
                qemb = self.modifier(qemb, qfemb) # can be either first q or modified q
            else:
                qfemb = self.qf_encoder(q_tokens[i], q_masks[i]).emb  # B H
                qemb = self.modifier(qembs[-1], qfemb) # can be either first q or modified q
            qembs.append(qemb)
        qembs = torch.stack(qembs, dim=1) # B N_seg H

        # loss calculation
        scores, loss_r = None, 0.0
        CELoss = nn.CrossEntropyLoss()

        # encode document if using contrastive signals
        dembs = []
        if (d_tokens is not None):
            n_candidates = (self.n_candidates or len(d_tokens))
            for i in range(n_candidates):
                demb = self.d_encoder(d_tokens[i], d_masks[i]).emb  # B H
                dembs.append(demb)
            dembs = torch.stack(dembs, dim=1) # B N_cand H

            scores = (qembs[:, 0, :]/self.tau) @ (dembs[:, 0, :]).T
            # scores = torch.einsum("ind, jd->inj", qembs[:, 0, :]/self.tau, dembs[:, 0, :]) # N B B
            labels = torch.arange(0, batch_size, dtype=torch.long, device=qembs.device)
            loss_r = CELoss(scores, labels) # first query and document

        return BiencoderOutput(
            qembs=qembs,
            dembs=dembs,
            loss=loss_r,
            scores=scores, 
            logs={'InfoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.qr_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.qf_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.gradient_checkpointing_enable(**kwargs)

    def get_encoder(self):
        return self.qf_encoder, self.qf_encoder
