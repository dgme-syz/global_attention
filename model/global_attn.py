from transformers import MODEL_MAPPING, AutoConfig, MODEL_FOR_CAUSAL_LM_MAPPING
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

class CustomAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor, # [batch_size, num_layers, hidden_size]
        value_states: torch.Tensor, # [batch_size, num_layers, seq_len, hidden_size]
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states.view(bsz, q_len, -1))
        attn_output = attn_output.view(bsz, q_len, -1, _)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


def get_ga_model(
    pretrained_model_name_or_path: str,
    **kwargs
):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    T = MODEL_MAPPING._load_attr_from_module(
        config.model_type, MODEL_MAPPING._model_mapping[config.model_type]
    )
    U = MODEL_FOR_CAUSAL_LM_MAPPING._load_attr_from_module(
        config.model_type, MODEL_FOR_CAUSAL_LM_MAPPING._model_mapping[config.model_type]
    )
    
    class GAModel(T):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.global_attn = CustomAttention(config=config, layer_idx=0) # train

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = True,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            
            if not output_hidden_states:
                Warning("output_hidden_states is forced to be True")
                config.output_hidden_states = output_hidden_states = True
            
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            
            all_layer_outputs = outputs.hidden_states[1:] # pop only embedding layer
            ##### like cls model, we select the last token (excpet pad) for query, key
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(outputs[0].device)
                else:
                    sequence_lengths = -1

            all_layer_outputs = torch.stack(all_layer_outputs)
            all_layer_outputs[:-1] = self.norm(all_layer_outputs[:-1]) # use the same setting as the original model
            rg = torch.arange(batch_size, device=all_layer_outputs.device)
            key_hidden_states = all_layer_outputs[:, rg, sequence_lengths, :] # [num_layers, batch_size, hidden_size]
            key_hidden_states = key_hidden_states.transpose(0, 1) # [batch_size, num_layers, hidden_size]
            all_layer_outputs = all_layer_outputs.transpose(0, 1) # [batch_size, num_layers, seq_len, hidden_size]
            attn_output, _ = self.global_attn(
                hidden_states=key_hidden_states,
                value_states=all_layer_outputs,
            )
            assert len(attn_output.shape) == 4
            attn_output = torch.mean(attn_output, dim=1) # [batch_size, seq_len, hidden_size]
            outputs.last_hidden_state = outputs.last_hidden_state + attn_output
            ##### global attention
            
            return outputs
            
    
    class GAForCausalLM(U):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.model = GAModel(config, **kwargs)
            
    
    return GAModel, GAForCausalLM