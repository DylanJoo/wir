import torch
import torch.nn as nn
from transformers import BertForMaskedLM, AutoConfig
from encoder.outputs import SparseEncoderOutput, DenseEncoderOutput

class SparseEncoder(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        head_mask=None,
        sub_input_ids=None,
        sub_attention_mask=None,
        sub_token_type_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        context_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        if sub_input_ids is not None:
            encoder_hidden_states = self.bert.embeddings(
                input_ids=sub_input_ids,
                token_type_ids=sub_token_type_ids,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0
            )
            encoder_attention_mask = sub_attention_mask

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        last_hidden_states = outputs[0]
        logits = self.cls(last_hidden_states)
        logits = logits * (context_mask or attention_mask).unsqueeze(-1)

        # pooling/aggregation
        values, _ = torch.max(
            torch.log(1 + torch.relu(logits)) 
            * attention_mask.unsqueeze(-1), dim=1
        )

        # normalization (for cos)
        # if self.norm:
        #     values = normalize(values)

        return SparseEncoderOutput(
            reps=values, 
            logits=logits, 
            last_hidden_states=last_hidden_states, 
            all_hidden_states=outputs["hidden_states"], 
            mask=attention_mask
        )

