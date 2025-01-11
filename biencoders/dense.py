import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
import math

class RankingValueHead(nn.Module):
    """ estimate the value of entire ranking """

    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, 768)
        self.fc_2 = nn.Linear(768, 1)
        summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.0)
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob != 0.0 else nn.Identity()

    def forward(self, logits):
        output = self.dropout(logits)
        if output.dtype != self.fc_1.weight.dtype:
            output = output.to(self.fc_1.weight.dtype)
        # output = self.summary(output).squeeze(-1)
        output = self.fc_2(self.fc_1(output))
        return output

class DenseAdaptiveReranker(nn.Module):

    def __init__(
        self, 
        opt, 
        q_encoder, 
        d_encoder=None,
        n_max_candidates=10,
        do_contrastive=False,
    ):
        super().__init__()
        self.opt = opt
        self.q_encoder = q_encoder
        self.d_encoder = d_encoder
        self.is_ddp = dist.is_initialized()
        self.tau = opt.tau
        self.vhead = RankingValueHead(input_size=n_max_candidates)
        self.n_max_candidates = n_max_candidates
        self.do_contrastive = do_contrastive

        self.scaling = math.sqrt(self.q_encoder.model.config.hidden_size)

        for n, p in self.named_parameters():
            if 'd_encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(
        self, 
        q_tokens, q_mask, 
        d_tokens, d_mask, 
        **kwargs
    ):
        loss, loss_r = 0.0, 0.0
        n_segments = len(q_tokens)
        n_candidates = len(d_tokens)
        batch_size = q_tokens[0].size(0)
        include_n_feedbacks = kwargs.pop('include_n_feedbacks', n_segments)

        qembs = self.q_encoder(q_tokens, q_mask, include_n_feedbacks).last_hidden_state  # B N_seg H

        dembs = []
        for i in range(n_candidates):
            demb = self.d_encoder(d_tokens[i], d_mask[i]).emb  # B H
            dembs.append(demb)
        dembs = torch.stack(dembs, dim=1) # B N_cand H

        ## 1) conetxt list-wise ranking for b-th batch
        ### ranking (candidates) <- N_seg N_cand
        # alpha = 0
        # r_ranking = 1/(alpha + 1 + (-score).argsort(-1)) # reciprocal

        all_scores = qembs @ dembs.mT # B N_seg N_cand
        # scores = scores / self.scaling
        logits = scores = torch.max(all_scores, 1).values # B N_cand
        ranking = (-scores).argsort(-1) # B N_cand

        ### mode1: max pooling over segmentes
        ### mode2: max pooling over element-wise product
        # all_logits = torch.einsum('ijk,ihk->ijhk', qembs, dembs) # B N_seg N_cand H
        # logits = all_logits.max(1).values # B N_cand H

        ## 2) constrastive learning
        if self.do_contrastive:
            qemb_ibn = qembs[:, 0, :] # the first segment (B H)
            demb_ibn = dembs[:, 0, :] # 

            # if self.is_ddp:
            #     gather_fn = gather
            #     demb_ibn = gather_fn(demb_ibn)

            labels = torch.arange(0, batch_size, dtype=torch.long, device=qemb_ibn.device)
            rel_scores = torch.einsum("id, jd->ij", qemb_ibn/self.tau, demb_ibn)
            CELoss = nn.CrossEntropyLoss()
            loss_r = CELoss(rel_scores, labels)

        return DenseEncoderOutput(
            qembs=qembs,
            loss=loss_r,
            logits=logits, # B N_cand H
            all_scores=all_scores, # B N_cand
            ranking=ranking,
            logs={'infoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.gradient_checkpointing_enable(**kwargs)

    def get_encoder(self):
        return self.q_encoder, None

