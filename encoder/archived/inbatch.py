import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from .dist_utils import gather

@dataclass
class InBatchOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class InBatchInteraction(nn.Module):

    def __init__(
        self, 
        opt, 
        q_encoder, 
        d_encoder=None,
    ):
        super().__init__()
        self.opt = opt
        self.q_encoder = q_encoder
        self.d_encoder = d_encoder
        self.is_ddp = dist.is_initialized()
        self.tau = opt.tau

        ## negative miner
        self.miner = miner
        self.n_negative_samples = opt.n_negative_samples

    def forward(
        self, 
        q_tokens, q_mask, 
        d_tokens, d_mask, 
        **kwargs
    ):
        loss, loss_r = 0.0, 0.0
        n_candidates = len(d_tokens[0])
        n_segments = len(q_tokens[0])

        qembs = self.q_encoder(q_tokens, q_mask).last_hidden_state  # B N_seg H

        dembs = []
        for i in range(n_segments):
            demb = self.d_encoder(d_tokens[i], d_mask[i]).emb  # B H
            dembs.append(demb)
        dembs = torch.stack(dembs, dim=1) # B N_cand H

        ## 1) conetxt list-wise ranking for b-th batch
        ### q <- qembs[b, :, :] N_seg H 
        ### d <- dembs[b, :, :] N_cand H 
        ### ranking (candidates) <- N_seg N_cand
        alpha = 0
        # for b in range(batch_size):
        #     score = qembs[b] @ demb[b].T # (N_seg H) x (N_cand H)
        #     r_ranking = 1/(alpha + 1 + (-score).argsort(-1)) # reciprocal
        #     print(r_ranking)
        #     reranking.append(r_ranking.sum(-2)) # N_seg x N_cand
        # print(reranking)
        # reranking = torch.stack(reranking, dim=0)

        sim_scores = qembs @ dembs.mT # (B N_seg H) x (B N_cand H) = B N_seg N_cand
        sim_scores = torch.max(sim_scores, 1).values

        ## mode1: esemble scores
        # sorted_items = torch.sort(pooled_socres, descending=True) 
        # ranking = sorted_items.indices

        ## mode1: reciprocal
        # ranking_scores = 1/(alpha + 1 + (-scores).argsort(-1)) 
        # reranking = ranking_scores.sum(-2).argsort(-1) # B N_cand

        ## 2) constrastive learning
        ### q <- qembs[:, 0, :] B (1) H. the first segment
        ### d <- dembs[:, 0, :] B (1) H. the first context (would change)
        ### scores (constrastive) <- B B 
        qemb_ibn = qembs[:, 0, :] # B H
        demb_ibn = dembs[:, 0, :] # B H
        ## [NOTE] here can also add negative by selecting bottom doc in cand

        if self.is_ddp:
            gather_fn = gather
            demb_ibn = gather_fn(demb_ibn)

        labels = torch.arange(
            0, qemb_ibn.size(0), 
            dtype=torch.long, 
            device=qemb_ibn.device
        )

        rel_scores = torch.einsum("id, jd->ij", qemb_ibn/self.tau, demb_ibn)

        ## computing losses
        CELoss = nn.CrossEntropyLoss()
        loss_ret = CELoss(rel_scores, labels)
        logs = {'infoNCE': loss_ret}

        return InBatchOutput(
            loss=loss_ret,
            scores=sim_scores,
            logs=logs, 
        )

    def get_encoder(self):
        return self.q_encoder, None

