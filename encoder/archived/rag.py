"""
# computing additional alignment loss
KLLoss = nn.KLDivLoss(reduction='batchmean')
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from typing import Optional, Union, List, Dict, Tuple, Any
from torch.nn import CrossEntropyLoss

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from prompt.qampari import *

@dataclass
class RAGOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    loss_r: torch.FloatTensor = None
    loss_g: torch.FloatTensor = None
    loss_kl: torch.FloatTensor = None
    answers: Optional[str] = None
    prompts_fbk: Optional[str] = None
    feedbacks: Optional[str] = None

class RerankAugmentedGeneration(nn.Module):

    def __init__(self, llm, tokenizer, biencoders=None, stop_token_ids=[], k=5):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.biencoders = biencoders # could be inbatch
        self.stop_token_ids = stop_token_ids # could be inbatch
        self.k = k

        # freeze G and R's d-encoder
        for p in self.llm.parameters():
            p.requires_grad = False

        if biencoders is not None:
            for p in self.biencoders.d_encoder.parameters():
                p.requires_grad = False

    def forward(
        self, 
        questions: List[str],
        targets: List[str] = None,
        candidates: Optional[List[List[Dict]]] = None,
        inputs_for_retriever: Optional[Dict] = None,
        **kwargs
    ):
        """
        params for generator
        --------------------
        questions: [G] the initial questions.
        targets: [G] the target answer (labels).
        candidates: [G] the candidate passages (context at t-1).
        k: [G] the depth of considered context after adaptive retrieval/re-ranking.

        params for retriever
        --------------------
        inputs_for_retrieverk: [G] the depth of considered context 
        after adaptive retrieval/re-ranking.

        returns
        -------
        """
        loss, loss_r, loss_g = 0.0, 0.0, 0.0

        ## step1. prepare the context via retrieve/rerank documents
        ## [NOTE] this could be move the `train.py`
        if inputs_for_retriever is not None:
            #### reranking via bi-encoders
            output_r = self.biencoders(**inputs_for_retriever)
            ## reorder/select documents
            contexts = []
            for candidate, ranking in zip(candidates, output_r.reranking):
                context = [p for _, p in sorted(zip(ranking, candidate))]
                contexts.append(context[:self.k])

            # [TODO] add post-G actions 
            # retrieved_pids = dosomething(output_r.scores)
            # updated_pids = dosomething(retrieved_pids, pids)
            # passages = [corpus[pid] for pid in updated_pids]
        else:
            contexts = [ctx[:self.k] for ctx in candidates]

        ## step2. prepare prompt of context for generation
        ## [TODO] cleaner to move this section to another function
        prompts = []
        prompts_fbk = []
        for i in range(len(questions)):
            D = apply_docs_prompt(contexts[i], field='text')
            # prompt for answering
            prompt = apply_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=instruction_prompt,
                prefix="Answer:"
            ).replace('{DEMO}', '')
            prompts.append(prompt)

            # prompt for getting feedback
            prompt_fbk = apply_fbk_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=fbk_instruction_prompt,
                prefix=fbk_prefix
            )
            prompts_fbk.append(prompt_fbk)

        ### Prepare source target
        inputs = self.tokenizer([f"{prompt} {target}" \
            for (prompt, target) in zip(prompts, targets)],
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(self.llm.device)

        #### get source length and revise labels
        labels = inputs['input_ids'].clone()
        # [NOTE] adjust this to tasks. Some of thme could also be ignored
        labels[labels==self.tokenizer.convert_tokens_to_ids(",")] = -100 
        labels[labels==self.tokenizer.pad_token_id] = -100 # remove the padded tokens
        tokenized_prompt = self.tokenizer(prompts, truncation=True)['input_ids']
        source_len = [len(tokens) for tokens in tokenized_prompt]
        for i, s in enumerate(source_len):
            labels[i, :(s-1)] = -100

        ## step3. forward pass with inputs (prompt and target)
        output_g = self.llm(**inputs, labels=None)

        ## step4. generate feedbacks 
        inputs_fbk = self.tokenizer(
            prompts_fbk,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.llm.device)
        # print("\n".join(prompts_fbk))

        output_fbk = self.llm.generate(
            **inputs_fbk, 
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=32,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id
        )
        feedbacks = []
        for i, output in enumerate(output_fbk):
            feedback = self.tokenizer.decode(
                output[inputs_fbk['input_ids'][i].size(-1):], 
                skip_special_tokens=True
            )
            feedback = feedback.split('\n\n')[0]
            feedbacks.append(feedback)

        ## verbose. it's slow
        # output = self.llm.generate(**inputs)
        # print(self.tokenizer.batch_decode(output))

        loss_r = output_r.loss
        loss_g = self.compute_nll(output_g.logits, labels)
        logs = {'InfoNCE': loss_r, 'mle': loss_g.mean()}

        return RAGOutput(
            loss=loss_r+loss_g.mean(),
            loss_r=loss_r,
            loss_g=loss_g,
            loss_kl=None,
            answers=None,
            prompts_fbk=prompts_fbk,
            feedbacks=feedbacks
        )

    def compute_nll(self, logits, labels):
        ## extract the batch-wise mean
        batch_size, _, vocab_size = logits.shape
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(batch_size, -1).mean(-1)
        return loss

    def postprocess(self, x):
        x = x.split('\n\n')[0] 
        x = x.split('Question:')[0] 
        return x

    ## [TODO] cleaner to move this section to another function
    # def _prepare_inputs_for_generator(self, contexts):
    #     prompts = []
    #     prompts_fbk = []
    #     for i in range(len(question)):
    #         D = apply_docs_prompt(contexts[i], field='text')
    #         # prompt for answering
    #         prompt = apply_inst_prompt(
    #             Q=question[i], 
    #             D=D,
    #             instruction=instruction_prompt,
    #             add_prefix=True
    #         ).replace('{DEMO}', '')
    #         prompts.append(prompt)
    #
    #         # prompt for getting feedback
    #         prompt_fbk = apply_fbk_inst_prompt(
    #             Q=question[i], 
    #             D=D,
    #             instruction=fbk_instruction_prompt,
    #             prefix=fbk_prefix
    #         )
    #         prompts_fbk.append(prompt_fbk)
    #     return prompts_fbk

