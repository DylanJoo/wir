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

@dataclass
class AdaptiveHeadOutput(BaseModelOutput):
    actions: torch.FloatTensor = None
    logprobs: List[torch.FloatTensor] = None
    values: List[torch.FloatTensor] = None
    loss_sft: torch.FloatTensor = None
    output: Optional[SparseEncoderOutput] = None

@dataclass
class DenseAdaptiveEncoderOutput(BaseModelOutput):
    qembs: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    logits: Optional[torch.FloatTensor] = None
    all_scores: Optional[torch.FloatTensor] = None
    ranking: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

@dataclass
class SparseAdaptiveEncoderOutput(BaseModelOutput):
    reps: torch.FloatTensor = None
    logprobs: torch.FloatTensor = None
    actions: Optional[torch.FloatTensor] = None
    prev_out: Optional[SparseEncoderOutput] = None
    q_out: Optional[SparseEncoderOutput] = None
    d_reps: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    loss_ct: torch.FloatTensor = None
    loss_mr: torch.FloatTensor = None
    loss_flop: torch.FloatTensor = None
    loss_tc: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
