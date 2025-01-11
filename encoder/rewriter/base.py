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
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class WriterOutput(BaseModelOutput):
    qembs: torch.FloatTensor = None
    dembs: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class FeedbackQueryRewriter(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            rewriter, # change it to "rewriter" later
            device_map='auto',
            attn_implmentation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(rewriter_name_or_path)
        self.tokenzier.padding_size = "left"

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
