import torch
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM, AutoConfig
from modeling.outputs import SparseEncoderOutput, DenseEncoderOutput
from modeling.layers import CrossAttentionLayer

def normalize(tensor, eps=1e-9):
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

class SparseEncoder(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()

        self.add_cross_attention = kwargs.pop('cross_attention', False)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)

        if self.add_cross_attention:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.crossattention_layers = nn.ModuleList(
                [CrossAttentionLayer(config, zero_init=False, mono_attend=False, output_layer=False) for _ in range(1)]
            )
            self.crossattention_cls = nn.Linear(self.model.config.hidden_size, 2)

        self.output = kwargs.pop('output', 'MLM')
        self.agg = kwargs.pop('agg', 'max')
        self.activation = kwargs.pop('activation', 'relu') 
        self.norm = kwargs.pop('norm', False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        context_mask=None,
    ):

        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        if (self.add_cross_attention) and (encoder_hidden_states is not None):
            last_hidden_states = outputs[0]

            for i, layer_module in enumerate(self.crossattention_layers):

                # option0: query as candidate
                # last_hidden_states = layer_module(
                #     hidden_states=last_hidden_states,
                #     attention_mask=attention_mask, 
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask, 
                # )[0]

                # option1: query+feedback as candidate
                candidate_hidden = torch.cat([last_hidden_states, encoder_hidden_states], 1)
                candidate_mask = torch.cat([attention_mask, encoder_attention_mask], 1)
                last_hidden_states = layer_module(
                    hidden_states=candidate_hidden,
                    attention_mask=candidate_mask,
                    encoder_hidden_states=candidate_hidden,
                    encoder_attention_mask=candidate_mask, 
                )[0]
                    # encoder_hidden_states=last_hidden_states,
                    # encoder_attention_mask=attention_mask, 

                # option2: feedback as candidate
                # last_hidden_states = layer_module(
                #     hidden_states=encoder_hidden_states,
                #     attention_mask=encoder_attention_mask,
                #     encoder_hidden_states=last_hidden_states,
                #     encoder_attention_mask=attention_mask, 
                # )[0]

            tok_logits = self.crossattention_cls(last_hidden_states)
            nonzero_indices = None

        else: 
            last_hidden_states = outputs[0]
            logits = self.model.cls(last_hidden_states)
            logits = logits * (context_mask or attention_mask).unsqueeze(-1)
            values, _ = torch.max(
                torch.log(1 + torch.relu(logits)) 
                * attention_mask.unsqueeze(-1), dim=1
            )

            tok_logits = None
            nonzero_indices = [row.nonzero(as_tuple=False).squeeze(1) for row in values]

        return SparseEncoderOutput(
            logits=tok_logits,
            indices=nonzero_indices,
            last_hidden_states=last_hidden_states, 
            all_hidden_states=outputs["hidden_states"], 
            mask=attention_mask
        )
