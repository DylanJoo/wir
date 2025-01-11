import numpy as np
import torch
import torch.nn as nn

EPSILON = np.finfo(np.float32).tiny

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        device = scores.device
        dtype = scores.dtype
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores, dtype=dtype)
        onehot_approx = torch.zeros_like(scores, dtype=dtype)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).to(device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx.to(device, dtype=dtype)

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot, dtype=dtype)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

def sample_actions(logits, samples=1, attention_mask=None):
    actions, logprobs = [], []
    probs = logits.softmax(-1)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)

    for i in range(samples):
        if i == (samples - 1): 
            action = torch.zeros_like(logits).scatter_(2, logits.argmax(-1).unsqueeze(-1), 1.)
            action = action.type(logits.dtype)
        else:
            action = m.sample()

        actions.append(action)
        logprob = m.log_prob(action).mean(-1)
        logprobs.append(logprob)
    return actions, logprobs

def multiple_sample_and_log_probability(
    scores, 
    sample_size, 
    return_prob=True, 
    batch=False,
    sort=False,
    baseline=False,
    tau=1
):
    if not batch:
        assert scores.dim() == 1
        subtracts = scores.new_zeros((sample_size, scores.size(0)))
        batch_index = torch.arange(sample_size, device=scores.device)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(0)):
            probs = nn.functional.softmax( (scores - subtracts)/tau, dim=1) + 1e-10
            if sort:
                posj = torch.argmax(probs, 1).squeeze(-1)
            elif baseline:
                posj = j
            else:
                posj = torch.multinomial(probs, 1).squeeze(-1)
            rankings.append(posj)
            if return_prob:
                log_probs[:, j] = probs[batch_index, posj].log()
            subtracts[batch_index, posj] = scores[posj] + 1e6
        rankings = torch.stack(rankings, dim=1)
        if return_prob:
            log_probs = log_probs.sum(dim=1)
            return rankings, log_probs
        else:
            return rankings

    else:
        assert scores.dim() == 2
        batch_size, candidiate_size = scores.size(0), scores.size(1)
        subtracts = scores.new_zeros((batch_size, sample_size, candidiate_size))
        batch_index = torch.arange(
            batch_size, device=scores.device).unsqueeze(1).expand(
            batch_size, sample_size)
        sample_index = torch.arange(
            sample_size, device=scores.device).expand(
            batch_size, sample_size)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(1)):
            probs = nn.functional.softmax(
                (scores.unsqueeze(1) - subtracts)/tau, dim=-1) + 1e-10
            if sort:
                posj = torch.argmax(
                    probs.reshape(batch_size * sample_size, -1),
                    1
                ).squeeze(-1).reshape(batch_size, sample_size)
            elif baseline:
                posj = torch.tensor(
                    [j] * (batch_size * sample_size)
                ).reshape(batch_size, sample_size)
            else:
                posj = torch.multinomial(
                    probs.reshape(batch_size * sample_size, -1),
                    1
                ).squeeze(-1).reshape(batch_size, sample_size)
            rankings.append(posj)
            if return_prob:
                log_probs[:, :, j] = probs[batch_index,
                                           sample_index, posj].log()
            subtracts[batch_index, sample_index,
                      posj] = scores[batch_index, posj] + 1e6
        rankings = torch.stack(rankings, dim=-1)
        if return_prob:
            log_probs = log_probs.sum(dim=-1)
            return rankings, log_probs
        else:
            return rankings
