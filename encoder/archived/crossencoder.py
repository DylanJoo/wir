import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class ValueOutput(BaseModelOutput):
    logits: torch.FloatTensor = None
    logprobs: torch.FloatTensor = None
    scores: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class ValueCrossEncoder(nn.Module):

    def __init__(
        self, 
        opt, 
        cross_encoder,
        d_encoder,
        n_max_candidates=None,
    ):
        super().__init__()
        self.opt = opt
        # BERTForSequenceClassification
        self.cross_encoder = cross_encoder
        self.d_encoder = d_encoder 
        self.n_max_candidates = n_max_candidates

    @staticmethod
    def _maybe_reshape(x):
        if x.dim() != 3:
            x = x[:, None, :]
        return x

    def _prepare_inputs(self, qemb, dembs):
        """ listwise re-ranker using embeddings to represent piece of texts """
        # prepare special tokens
        cls, sep, device = 101, 102, self.cross_encoder.device
        # size: [1, 2]
        embeds = self.cross_encoder.bert.embeddings(
            torch.tensor([[cls, sep]]).to(device).repeat( (qemb.size(0), 1) )
        )
        cls_emb = embeds[:, 0:1]
        sep_emb = embeds[:, 1:2]

        # prepare text embeddings
        qemb = self._maybe_reshape(qemb) # B N H
        dembs = self._maybe_reshape(dembs) # B M H
        dembs = dembs[:, :self.n_max_candidates, :]
        # print(cls_emb.shape)
        # print(sep_emb.shape)
        # print(qemb.shape)
        # print(dembs.shape)

        # concat everything
        embeds = torch.cat([cls_emb, qemb, sep_emb, dembs, sep_emb], axis=1)
        token_type_ids = torch.ones( (embeds.size(0), embeds.size(1)) , dtype=torch.long)
        token_type_ids[:, :(embeds.size(1))] = 0

        return {'input_ids': None,
                'inputs_embeds': embeds,
                'attention_mask': None,
                'token_type_ids': token_type_ids.to(device),
                'output_hidden_states': True}

    ## [NOTE] Rewards ~ r = CrossEncoder(E_qr, E_qf)
    def forward(self, qemb, dembs, **kwargs):
        ## [CLS] <e_q> [SEP] <e_d1> <e_d2> ... [SEP]
        loss = 0.0
        batch_size = qemb.size(0)

        # encode candidates
        # B = 3 | N = 4 | M = 2
        inputs = self._prepare_inputs(qemb, dembs) # (B (1) H) x (B M H)
        model_output = self.cross_encoder(**inputs, **kwargs)

        # print(logits.shape)
        if self.cross_encoder.config.num_labels == 1:
            scores = model_output.logits.squeeze()
        else:
            scores = torch.nn.functional.softmax(model_output.logits, dim=-1) 
            logprobs = torch.log(scores)
            labels = torch.argmax(logprobs, -1)[:, None]

        return ValueOutput(
            logits=model_output.logits,
            logprobs=logprobs,
            labels=labels,
            scores=scores,
            last_hidden_states=model_output.hidden_states[-1], # this means nothing unless you check all
            loss=loss,
            logs={}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.cross_encoder.model.gradient_checkpointing_enable(**kwargs)

