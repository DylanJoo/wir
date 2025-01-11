import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="naver/splade-v3")
    add_pooling_layer: Optional[bool] = field(default=False)
    num_mem_tokens: Optional[int] = field(default=1)
    num_budget: Optional[int] = field(default=5)
    tau: Optional[float] = field(default=1.0)

from base_encoder import SparseEncoder
from biencoders import SparseAdaptiveEncoders
from biencoders.sparse import RegularizationHead

model_opt = ModelOptions(retriever_name_or_path='naver/splade-v3')
tokenizer = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
def test_se():
    modifier = RegularizationHead(
        model_opt,
        encoder=SparseEncoder(
            model_name_or_path=model_opt.retriever_name_or_path,
            output='MLM', agg='max', activation='relu'
        )
    )
    ada_retriever = SparseAdaptiveEncoders(
        model_opt,
        encoder=SparseEncoder(model_name_or_path=model_opt.retriever_name_or_path).eval(),
        modifier=modifier
    )

    input1 = tokenizer(['Who played prudence on nanny and the professor?'], return_tensors='pt')
    # input2 = tokenizer(['hello world'], return_tensors='pt')

    values, logprobs, actions = modifier(input1['input_ids'], input1['attention_mask'])
    print(logprobs)
    print(logprobs.nonzero().shape)
    print(tokenizer.batch_decode(torch.argsort(values, -1, descending=True)[:, :8]))

# test_biencder()
# test_crossencoder()
# test_reward_wrapper()
test_se()

# Actionvalue ['nannyudence professor actor pr she nan character', 'pr nannyudence professor character tv she nan', 'pr nannyudence professor actor she character nan', 'professoryn dawn everett butch nannyudence l']
