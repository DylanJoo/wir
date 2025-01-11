import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from base_encoder import SparseEncoder

class SparseRetrievalRL(nn.Module):
    def __init__(self, opt, sampling_method='gumbel'):
        super().__init__()
        self.sampling_method = sampling_method
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = SparseEncoder(opt.generator_name_or_path)
        config = self.encoder.model.config
        
        # Policy network to learn discretization
        self.policy_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2)  # Binary decision for each dimension
        )
        
        # Value network for critic
        self.value_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        
    def sample_policy(self, logits, temperature=1.0):
        if self.sampling_method == 'gumbel':
            # Gumbel-Softmax sampling
            policy = F.gumbel_softmax(logits, tau=temperature, hard=True)
            log_probs = F.log_softmax(logits, dim=-1)
            return policy, log_probs
        
        elif self.sampling_method == 'categorical':
            # Explicit categorical sampling
            probs = F.softmax(logits / temperature, dim=-1)
            m = Categorical(probs)
            actions = m.sample()
            log_probs = m.log_prob(actions)
            
            # Convert to one-hot for sparse representation
            policy = F.one_hot(actions, num_classes=2).float()
            return policy, log_probs
    
    def forward(self, inputs, temperature=1.0):
        # Get embeddings and encode
        q_out = self.encoder(inputs['q_tokens'], inputs['q_masks'])
        pooled = torch.max(q_out.rep, dim=1)[0]
        
        # Get policy logits and value estimate
        policy_logits = self.policy_net(q_out.last_hidden_states)
        value = self.value_net(pooled)
        
        # Sample actions using specified method
        policy, log_probs = self.sample_policy(policy_logits, temperature)
        
        # Create sparse representation
        if self.sampling_method == 'categorical':
            sparse_repr = pooled * (policy == 1).float()
        else:
            sparse_repr = pooled * policy[:, :, 1].unsqueeze(-1)
        
        return {
            'sparse_repr': sparse_repr,
            'policy_logits': policy_logits,
            'log_probs': log_probs,
            'value': value,
            'policy': policy
        }
    
    def compute_loss(self, batch, rewards, temperature=1.0, gamma=0.99):
        outputs = self.forward(batch, temperature)
        
        if self.sampling_method == 'gumbel':
            # Gumbel-Softmax version
            log_probs = outputs['log_probs']
            selected_log_probs = log_probs[:, :, 1].mean(dim=1)  # For active dimensions
        else:
            # Categorical sampling version
            selected_log_probs = outputs['log_probs']
        
        advantages = rewards - outputs['value'].detach()
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(outputs['value'], rewards)
        
        # Sparsity regularization
        sparsity_penalty = torch.norm(outputs['sparse_repr'], p=1)
        
        # Entropy regularization for exploration
        if self.sampling_method == 'categorical':
            probs = F.softmax(outputs['policy_logits'], dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            entropy_bonus = 0.01 * entropy
        else:
            entropy_bonus = 0
        
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * sparsity_penalty - entropy_bonus
        
        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'sparsity': sparsity_penalty.item(),
            'entropy': entropy_bonus.item() if self.sampling_method == 'categorical' else 0
        }


# Example usage
def train_step(model, optimizer, batch, rewards):
    optimizer.zero_grad()
    
    # Anneal temperature over training
    temperature = max(0.5, 1.0 - epoch * 0.1)  # Example annealing schedule
    
    loss, metrics = model.compute_loss(batch, rewards, temperature)
    loss.backward()
    optimizer.step()
    
    return metrics
