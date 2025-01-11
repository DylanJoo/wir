import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import random
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from modeling.biencoders.layers import CrossAttentionLayer
from modeling.utils import SubsetOperator
from modeling.utils import multiple_sample_and_log_probability

class AttentionTopkHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        self.q_encoder = encoder
        self.gumbel_topk = SubsetOperator(k=10, hard=True)
        config = AutoConfig.from_pretrained(opt.retriever_name_or_path)
        self.attn_layer = CrossAttentionLayer(config)
        self.samples = opt.samples

    def forward(self, input_ids, attention_mask, q_out=None):
        ## query hidden
        q_embeds = q_out.last_hidden_state

        ## feedback
        out = self.q_encoder(input_ids, attention_mask)
        f_embeds = out.last_hidden_state

        ## cross-attention: output --> B N_heads Lq Lf --> B Lq Lf
        attention_scores = self.attn_layer(
            hidden_states=q_embeds, encoder_hidden_states=f_embeds,
            output_attention_scores=True
        )[1]
        attention_scores = attention_scores.mean(1)

        ## maximum feedback-token including score
        aggregated_scores = torch.max(attention_scores, dim=1).values

        all_logprobs = F.log_softmax(aggregated_scores, dim=-1)
        probs = all_logprobs.exp()
        logprobs, actions, values = [], [], []

        ## Opt1: sampling
        # print('\nprob (pos)', (probs>=0.5).sum(-1) / probs.shape[-1])
        # print('\nprob (neg)', (probs<0.5).sum(-1) / probs.shape[-1])
        # print('\nprob (max)', probs.max(-1).values)
        # actions:  [ (B Lf), (B Lf), ...]
        # values:   [ (B V), (B V), ...]
        # logprob:  [ (B), (B), ...]
        actions, values, logprobs = [], [], []
        for i in range(self.samples):
            action = self.gumbel_topk(aggregated_scores) # B Lf
            actions.append(action)

            logprob = (all_logprobs * action).sum(1) # B
            logprobs.append(logprob)

            old_logits = torch.max(q_out.logits, dim=1).values
            new_logits = torch.max(out.logits * action.unsqueeze(-1), dim=1).values
            value = torch.log(1 + torch.relu(old_logits + new_logits))
            values.append(value)
        return values, logprobs, actions


class AttentionHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        config = AutoConfig.from_pretrained(opt.retriever_name_or_path)
        self.attn_layer = CrossAttentionLayer(config)
        self.q_encoder = encoder
        self.samples = opt.samples

    def forward(self, input_ids, attention_mask, q_out=None):
        ## query hidden
        q_embeds = q_out.last_hidden_state

        ## feedback hidden
        out = self.q_encoder(input_ids, attention_mask)
        f_embeds = out.last_hidden_state

        ## cross-attention: output --> B N_heads Lq Lf --> B Lq Lf
        attention_scores = self.attn_layer(
            hidden_states=q_embeds, encoder_hidden_states=f_embeds,
            output_attention_scores=True
        )[1]
        attention_scores = attention_scores.mean(1)

        ## maximum feedback-token including score
        aggregated_scores = torch.max(attention_scores, dim=1).values

        ## transform into probability of actions 
        logprobs, actions, values = [], [], []

        ## Opt1: sampling
        probs = torch.sigmoid(aggregated_scores)
        probs = probs * attention_mask
        print('\nprob (pos)', (probs>=0.5).sum(-1) / probs.shape[-1])
        print('\nprob (neg)', (probs<0.5).sum(-1) / probs.shape[-1])
        # print('\nprob (max)', probs.max(-1).values)
        dist = torch.distributions.Bernoulli(probs) # B Lf
        ## actions:  [ (B Lf), (B Lf), ...]
        ## values:   [ (B V), (B V), ...]
        ## logprob:  [ (B), (B), ...]
        for i in range(self.samples):
            if i == (self.samples - 1): 
                # put the deterministic sample at the end
                action = (probs >= 0.5).clone()
                action = action.type(q_out.logits.dtype)
            else:
                action = dist.sample()
            actions.append(action)
            logprob = dist.log_prob(action).sum(-1)
            logprobs.append(logprob)

            # [opt1] combine them and do the pooling together
            aggregated_logits = torch.cat(
                [q_out.logits, out.logits * action.unsqueeze(-1)], 
                dim=1
            )
            value, _ = torch.max(
                torch.log(1 + torch.relu(aggregated_logits)) * 
                torch.cat([q_out.mask, attention_mask], dim=1).unsqueeze(-1), 
                dim=1
            )

            # [opt2] combine them before pooling feedback-aware logit
            # old_logits = torch.max(q_out.logits, dim=1).values
            # new_logits = torch.max(out.logits * action.unsqueeze(-1), dim=1).values
            # value = torch.log(1 + torch.relu(old_logits + new_logits))

            # [opt3] replace with feedback-aware logit
            # new_logits = out.logits * action.unsqueeze(-1)
            # value, _ = torch.max(
            #     torch.log(1 + torch.relu(new_logits)) * attention_mask.unsqueeze(-1),
            #     dim=1
            # )

            values.append(value)
        return values, logprobs, actions, q_out

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        opt, 
        encoder, 
        modifier=None,
        d_encoder=None,
        n_candidates=None
    ):
        super().__init__()
        self.opt = opt
        
        # modeling
        self.q_encoder = encoder
        self.d_encoder = encoder
        self.modifier = modifier
        self.tau = opt.tau
        self.n_candidates = n_candidates

        for n, p in self.named_parameters():
            if 'encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, q_tokens, q_masks, prev_out, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('include_n_feedbacks', n_segments)
        batch_size = q_tokens[0].size(0)
        q_reps = []
        q_logprobs = []
        q_actions = []

        # encode query feedback
        for i in range(1, max_num_steps+1): # [1, ..., max_num_steps]
            q_rep, q_logprob, q_action, q_out = self.modifier(q_tokens[i], q_masks[i], prev_out)
            q_reps += q_rep
            q_logprobs += q_logprob
            q_actions += q_action
            ## actions:  [ (B Lf), (B Lf), ...]
            ## values:   [ (B V), (B V), ...]
            ## logprob:  [ (B), (B), ...]

        q_reps = torch.stack(q_reps, dim=1) # B N V
        q_logprobs = torch.stack(q_logprobs, dim=1) # B N

        # loss calculation
        scores, loss_r = None, 0.0
        CELoss = nn.CrossEntropyLoss()

        # encode document if using contrastive signals
        d_reps = []
        if (d_tokens is not None):
            n_candidates = (self.n_candidates or len(d_tokens))
            for i in range(n_candidates):
                d_rep = self.d_encoder(d_tokens[i], d_masks[i]).reps # B H
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=1) # B N_cand H

            # B Nq V x B d+ V = B 
            # B Nq V x B d- V
            scores = (q_reps[:, -1, :]/self.tau) @ (d_reps[:, 0, :]).T # last query x positive context 
            labels = torch.arange(0, batch_size, dtype=torch.long, device=q_reps.device)
            loss_r = CELoss(scores, labels) # first query and document

        return EncodersOutput(
            q_reps=q_reps,
            q_logprobs=q_logprobs,
            q_actions=q_actions,
            q_out=q_out,
            d_reps=d_reps,
            loss=loss_r,
            scores=scores, 
            logs={'InfoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)

# deprecated
class AnsweringHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        self.encoder = encoder
        self.gumbel_topk = SubsetOperator(k=1000, hard=True)

    def forward(self, input_ids, attention_mask, query_rep=None):
        logits = self.encoder(input_ids, attention_mask).logits
        all_logprobs = F.log_softmax(logits, dim=-1)  # B L V
        batch_size, _, vocab_size = logits.size(0), logits.size(-1)
        selections = self.gumbel_topk(logits.view(-1, vocab_size))         # B L V
        selections = selections.view(batch_size, -1, vocab_size)
        logprobs = (all_logprobs * selections).sum(1) # B V
        values, _ = torch.max(
            torch.log(1 + torch.relu(logits * selections)) 
            * attention_mask.unsqueeze(-1), dim=1
        )
        return values, logprobs, selections


# class RegularizationHead(nn.Module):
#     def __init__(self, opt, encoder):
#         super().__init__()
#         self.encoder = encoder
#         self.samples = opt.samples
#
#     def forward(self, input_ids, attention_mask, rep=None):
#         out = self.encoder(input_ids, attention_mask)
#         value = F.softplus(out.logits)
#         dist = torch.distributions.Bernoulli(value * (rep.unsqueeze(1)>0) ) # B V 
#
#         logprobs, actions, values = [], [], []
#         for i in range(self.samples):
#             actions_ = dist.sample()
#             logprobs_ = dist.log_prob(actions_).sum(1)
#
#             logprobs.append(logprobs_.sum(-1))
#             actions.append(actions_.max(1).values )
#             values.append(rep * actions_.max(1).values ) 
#         return values, logprobs, actions

