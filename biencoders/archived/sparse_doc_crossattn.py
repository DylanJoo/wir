import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput
from modeling.utils import sample_actions

def make_labels(d_tokens, candidate_tokens, candidate_masks, q_tokens=None):
    binary_matrix = torch.zeros_like(candidate_tokens)
    for i in range(len(d_tokens)):
        binary_matrix[i] = (candidate_tokens[i].unsqueeze(1) == d_tokens[i]).any(dim=1)
        if q_tokens is not None:
            binary_matrix[i] = (candidate_tokens[i].unsqueeze(1) == q_tokens[i]).any(dim=1)

    # random sample negatives
    # rand_matrix = torch.randint(0, 2, binary_matrix.shape).to(candidate_tokens.device)
    # rand_matrix = rand_matrix * (binary_matrix == 0) 
    # binary_matrix = torch.where(rand_matrix==1, 0, -100)

    # mask the unused token
    mask_matrix = torch.full_like(binary_matrix, -100)
    binary_matrix = torch.where(candidate_masks==0, mask_matrix, binary_matrix)
    return binary_matrix.to(candidate_tokens.device)

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        q_encoder,
        encoder=None, 
        n_candidates=None,
        **kwargs # opt is unused
    ):
        super().__init__()
        self.q_encoder = q_encoder
        self.encoder = (encoder or q_encoder)
        self.n_candidates = n_candidates
        self.config = q_encoder.model.config

        for n, p in self.named_parameters():
            if 'crossattention' in n:
                p.requires_grad = True
                print(n)
            else:
                p.requires_grad = False

    def _flop(self, q_value):
        lambda_t_q = 0.005
        q_value = torch.sum(torch.mean(torch.abs(q_value), dim=0) ** 2)
        return q_value * lambda_t_q

    def get_contrastive_loss(self):
        batch_size, vocab_size = output.reps.shape
        CELoss = nn.CrossEntropyLoss()

        n_candidates = min(self.n_candidates, len(d_tokens))
        for i in range(n_candidates):
            d_rep = self.encoder(d_tokens[i], d_masks[i]).reps
            d_reps.append(d_rep)
        d_reps = torch.stack(d_reps, dim=0) # N_cand B H

        ## L0: flop
        loss_flop = self._flop(output.reps)

        ## L1: contrastive learning
        scores_t = output.reps @ d_reps.view(-1, vocab_size).transpose(1, 0)    # B NB
        labels_ct = torch.arange(0, batch_size, device=output.reps.device, dtype=torch.long)
        loss_ct = CELoss(scores_t, labels_ct)
        # scores_0 = prev_output.reps @ d_reps.view(-1, vocab_size).transpose(1, 0) # B B
        # loss_ct_baseline = CELoss(scores_0, labels)
        return loss_ct

    def forward(
        self, 
        q_tokens=None, q_masks=None,
        f_tokens=None, f_masks=None, 
        d_tokens=None, d_masks=None, 
        prev_output=None,
        step=0,
        **kwargs
    ):
        q_reps, d_reps = None, []
        loss_tc, loss_flop, loss_ct, loss_mr = None, None, None, None
        pos_ratio = 0
        logprob = None

        if (step == 0) and (prev_output is None):
            prev_output = output = self.encoder(q_tokens, q_masks)
            reps = q_tokens
        else:
            f_output = self.encoder(f_tokens, f_masks)
            output = self.q_encoder(
                q_tokens, 
                q_masks, 
                encoder_hidden_states=f_output.last_hidden_states,
                encoder_attention_mask=f_masks
            )

            # option1: query+feedback as candidate
            candidate_tokens = torch.cat([q_tokens, f_tokens], 1)
            candidate_masks = torch.cat([q_masks, f_masks], 1)

            # option2: feedback as candidate
            # candidate_tokens = f_tokens
            # candidate_masks = f_masks

            # selection = output.logits.softmax(-1)[:, :, 1]
            # print('selection', selection[0])
            action, logprob = sample_actions(output.logits, samples=2)
            print('action', action[0][0, :, 1])
            logprob = logprob[0]
            select_tokens = torch.where(
                action[0][:, :, 1]==1, candidate_tokens, torch.full_like(candidate_tokens, 0)
            )

            # reps = torch.cat([q_tokens, select_tokens], 1)
            reps = select_tokens

            batch_size, seq_size, _ = output.logits.shape
            CELoss = nn.CrossEntropyLoss()

            if d_tokens is not None:
                labels = []
                n_candidates = min(self.n_candidates, len(d_tokens))
                for i in range(n_candidates):
                    d_rep = self.encoder(d_tokens[i], d_masks[i]).indices

                    label = make_labels(
                        d_rep, candidate_tokens, candidate_masks, q_tokens
                    )
                    labels.append(label)

                ## L1: token classification
                loss_tc = CELoss(output.logits.view(-1, 2), labels[0].view(-1))
                pos_ratio = (labels[0]==1).sum()  / (labels[0]!=-100).sum()

                ## L2: contrastive learning
                # labels = torch.cat(labels, 0).float() # N_cand B H
                # scores = output.logits[:, :, 1] @ labels.view(-1, seq_size).permute(1, 0)
                # labels_ct = torch.arange(0, batch_size, device=output.logits.device, dtype=torch.long)
                # loss_ct = CELoss(scores, labels_ct)

        return SparseAdaptiveEncoderOutput(
            reps=reps,
            prev_out=output,
            d_reps=d_reps,
            loss_ct=torch.tensor([0.0]),
            loss_mr=torch.tensor([0.0]),
            loss_flop=torch.tensor([0.0]),
            loss_tc=loss_tc,
            logs={'InfoNCE': loss_ct, 'PosRatio': pos_ratio},
            logprobs=logprob,
            logits=output.logits,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
