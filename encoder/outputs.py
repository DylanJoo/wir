import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class DenseEncoderOutput(BaseModelOutput):
    reps: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class SparseEncoderOutput(BaseModelOutput):
    reps: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    weights: torch.FloatTensor = None
    indices: torch.FloatTensor = None
    mask: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    mlm_hidden_states: torch.FloatTensor = None
    all_hidden_states: torch.FloatTensor = None

