import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoConfig
from modeling.layers import CrossAttentionLayer
from modeling.outputs import AdaptiveHeadOutput, SparseAdaptiveEncoderOutput

class AttentionHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        config = AutoConfig.from_pretrained(opt.retriever_name_or_path)
        config.num_attention_heads = 12
        config.num_layers = 1

        self.config = config
        self.crossattention = nn.ModuleList([CrossAttentionLayer(config) for _ in range(config.num_layers)])
        self.encoder = encoder.eval()
        self.samples = opt.samples
        self.args = opt

    def forward(self, input_ids, attention_mask, q_out, ignore_value_projection=True):
        device = input_ids.device
        f_out = self.encoder(input_ids, attention_mask)

        ## query and feedback vectors
        q_logits = q_out.logits
        f_logits = f_out.logits

        ## Policy model: cross-attention
        for i, attn_layer in enumerate(self.attn_layer):
            if i == 0:
                attn_out = crossattention(
                    hidden_states=q_out.last_hidden_states, 
                    attention_mask=q_out.mask,
                    encoder_hidden_states=f_out.last_hidden_states,
                    encoder_attention_mask=f_out.mask,
                    output_attention_scores=True,
                    ignore_value_projection=ignore_value_projection
                )
            else:
                attn_out = crossattention(
                    hidden_states=attn_out[0], 
                    attention_mask=q_out.mask,
                    encoder_hidden_states=f_out.last_hidden_states,
                    encoder_attention_mask=f_out.mask,
                    output_attention_scores=True,
                    ignore_value_projection=ignore_value_projection
                )

        # B Lf Lq H  
        if self.config.num_attention_heads == 1:
            qf_attentions = attn_out[1].squeeze(1)
        else:
            qf_attentions = attn_out[1].max(1).values
        qf_logits = self.encoder.model.cls(attn_out[0])

        ## transform into probability of actions 
        values = []
        if self.samples > 1:
            actions, logprobs = self.sample_actions(states=qf_attentions, attention_mask=attention_mask)

            for action in actions:
                # deterministic
                value, _ = torch.max(torch.log(1 + torch.relu(q_out.logits + qf_logits)), dim=1)
                # sampled
                values.append(value)
        else:
            actions = []
            logprobs = [torch.tensor([0.0] * f_logits.size(0)).to(device)] 
            value = torch.max(torch.log(1 + torch.relu(qf_logits)), dim=1).values
            values.append(value)

        return AdaptiveHeadOutput(
            actions=actions,
            logprobs=logprobs,
            values=values,
            output=attn_out,
        )

    def sample_actions(self, states, attention_mask=None):
        actions, logprobs = [], []
        probs = states.softmax(-1)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)

        for i in range(self.samples):
            if i == (self.samples - 1): 
                action = torch.zeros_like(states).scatter_(2, states.argmax(-1).unsqueeze(-1), 1.)
                action = action.type(states.dtype)
            else:
                action = m.sample()

            actions.append(action)
            logprob = m.log_prob(action).sum(-1)
            logprobs.append(logprob)

            # [opt1] combine them and do the pooling together
            # aggregated_logits = torch.cat(
            #     [q_out.logits, out.logits * action.unsqueeze(-1)], 
            #     dim=1
            # )
            # value, _ = torch.max(
            #     torch.log(1 + torch.relu(aggregated_logits)) * 
            #     torch.cat([q_out.mask, attention_mask], dim=1).unsqueeze(-1), 
            #     dim=1
            # )
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
        return actions, logprobs

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
        self.d_encoder = encoder
        self.modifier = modifier
        self.tau = opt.tau
        self.n_candidates = n_candidates
        for n, p in self.named_parameters():
            if 'crossattention' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, q_tokens, q_masks, prev_out, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('include_n_feedbacks', n_segments)
        batch_size = q_tokens[0].size(0)
        q_reps = []
        q_logprobs = []
        q_actions = []

        # encode query feedback
        for i in range(0, max_num_steps+1): # [1, ..., max_num_steps]
            if i == 0:
                pass
            else:
                output = self.modifier(q_tokens[i], q_masks[i], prev_out, ignore_value_projection=True) 
                q_reps += output.values
                q_logprobs += output.logprobs
                q_actions += output.actions

        q_reps = torch.stack(q_reps, dim=1) # B N V
        q_logprobs = torch.stack(q_logprobs, dim=1) # B N

        # loss calculation
        scores, loss_ct = None, 0.0
        CELoss = nn.CrossEntropyLoss()
        MRLoss = nn.MarginRankingLoss()

        # encode document if using contrastive signals
        d_reps = []
        if d_tokens is not None:
            n_candidates = min(self.n_candidates, len(d_tokens))
            for i in range(n_candidates):
                d_rep = self.d_encoder(d_tokens[i], d_masks[i]).reps # B H
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=0) # N_cand B H

            ## merge from different sources 
            # B Nq V x B d+ V = B V x B V N --last query x positive context 
            d_reps_T = d_reps[0]
            d_reps_F = d_reps[1]
            scores_0 = prev_out.reps @ d_reps.view(-1, q_reps.size(-1)).permute(1, 0)
            scores_t = q_reps[:, -1, :] @ d_reps.view(-1, q_reps.size(-1)).permute(1, 0)
            labels = torch.arange(0, batch_size, dtype=torch.long, device=q_reps.device)
            loss_ct = CELoss(scores_t, labels) 

            margin_0 = prev_out.reps * d_reps_T - prev_out.reps * d_reps_F
            margin_t = q_reps[:, -1] * d_reps_T - q_reps[:, -1] * d_reps_F
            labels_mr = torch.ones(margin_0.shape, dtype=torch.long, device=q_reps.device)
            loss_mr = MRLoss(margin_t, margin_0, labels_mr).sum()

        return SparseAdaptiveEncoderOutput(
            reps=q_reps,
            logprobs=q_logprobs,
            actions=q_actions,
            out=output.output,
            d_reps=d_reps,
            loss_ct=loss_ct,
            loss_mr=loss_mr, 
            scores=scores, 
            logs={'InfoNCE': loss_ct}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)
