# import torch
# import torch.nn as nn
# from transformers import BertForMaskedLM, AutoConfig
# from encoder.outputs import SparseEncoderOutput, DenseEncoderOutput

## Learned dense encoder
class Contriever(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()

        self.add_cross_attention = kwargs.pop('cross_attention', False)
        self.model = BertModel.from_pretrained(model_name_or_path)

        if self.add_cross_attention:
            config = AutoConfig.from_pretrained(model_name_or_path)
            config.num_attention_heads = 1
            self.crossattentionlayer = CrossAttentionLayer(
                config, zero_init=False, mono_attend=False, output_layer=False
            )

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

        outputs = self.model.forward(
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

        if attention_mask is not None:
            last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if (self.add_cross_attention) and (encoder_hidden_states is not None):
            last_hidden_states = self.crossattentionlayer(
                hidden_states=outputs[0],
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )[0]
        else: 
            last_hidden_states = outputs[0]

        emb = last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return DenseEncoderOutput(
            reps=emb, 
            last_hidden_states=last_hidden_states
        )
