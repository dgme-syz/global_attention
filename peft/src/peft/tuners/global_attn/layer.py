import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer


from transformers import MODEL_MAPPING
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch.nn.functional as F

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
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        value_states = value_states.to(self.q_proj.weight.dtype)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        attn_output = F.scaled_dot_product_attention(
            query=query_states, key=key_states, value=value_states.view(bsz, q_len, -1), dropout_p=self.attention_dropout
        )
        attn_output = attn_output.view(bsz, q_len, -1, _)
        return attn_output.to(dtype)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = 2 * self.hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        dtype = hidden_state.dtype
        hidden_state = hidden_state.to(self.gate_proj.weight.dtype)
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)).to(dtype)

def get_ga_model(config: str):
    T = MODEL_MAPPING._load_attr_from_module(
        config.model_type, MODEL_MAPPING._model_mapping[config.model_type]
    )
    
    class GAModel(BaseTunerLayer, T):
        adapter_layer_names = ("global_attn", "global_mlp")
        
        def __init__(self, 
            config,
            base_model,
            adapter_name: str, 
            **kwargs):
            super().__init__(base_model.config)
            self.global_attn = nn.ModuleDict({}) # CustomAttention(config=config, layer_idx=0) # train
            self.global_mlp = nn.ModuleDict({}) # MLP(config)
            self.name = adapter_name
            self._disable_adapters = False
            self.merged_adapters = []
            self.update_layer(base_model, adapter_name)
            self.hooks = []
            self.layer_out = ()
            self.ga_config = config
            self.config = base_model.config
            self.set_hook()
            torch.cuda.empty_cache()
        
        def __repr__(self) -> str:
            return super().__repr__()
        
        def hook(self, module, inp, out):
            if not hasattr(module, "self_attn"):
                raise ValueError(f"Error Layer!")
            self.layer_out += (out[0], )
        
        def set_hook(self):
            num_layers = self.config.num_hidden_layers
            seq = list(range(1, num_layers - 1))
            if self.ga_config.target != "all":
                seq = seq[-int(self.ga_config.target):]
            
            for x in seq:
                self.hooks.append(self.layers[x].register_forward_hook(self.hook))
            print(
                f"-----------------Use the following layers for global attention-------------------\n{seq}"
            )
                
        def __del__(self):
            for hook in self.hooks:
                hook.remove()
            
        def update_layer(self, base_model, adapter_name):

            self.global_attn[adapter_name] = CustomAttention(base_model.config)
            self.global_mlp[adapter_name] = MLP(base_model.config)
            
            self.set_adapter(self.active_adapters)
            # initialize the adapter layer
            if adapter_name in self.global_attn:
                nn.init.xavier_normal_(self.global_attn[adapter_name].q_proj.weight.data)
                nn.init.xavier_normal_(self.global_attn[adapter_name].k_proj.weight.data)
            if adapter_name in self.global_mlp:
                nn.init.xavier_normal_(self.global_mlp[adapter_name].gate_proj.weight.data)
                nn.init.xavier_normal_(self.global_mlp[adapter_name].up_proj.weight.data)
                nn.init.xavier_normal_(self.global_mlp[adapter_name].down_proj.weight.data)
                
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
            output_attentions = False
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
            dtype = outputs.last_hidden_state.dtype
            with torch.no_grad():
                all_layer_outputs = self.layer_out
                all_layer_outputs = torch.stack(all_layer_outputs)
                key_hidden_states = torch.mean(all_layer_outputs, dim=-2).to(dtype)
                key_hidden_states = key_hidden_states.transpose(0, 1) # [batch_size, num_layers, hidden_size]
                all_layer_outputs = all_layer_outputs.transpose(0, 1) # [batch_size, num_layers, seq_len, hidden_size]
                residual = all_layer_outputs
            
            attn_output = self.global_attn[self.name](
                hidden_states=key_hidden_states,
                value_states=all_layer_outputs,
            )
            # a dangerous operation, but it is necessary to save memory
            del key_hidden_states, all_layer_outputs
            torch.cuda.empty_cache()
    
            assert len(attn_output.shape) == 4
            attn_output = attn_output + residual
            residual = attn_output
            attn_output = self.global_mlp[self.name](attn_output)
            
            attn_output = attn_output + residual
            hidden_states = attn_output.transpose(0, 1) # [num_layers, batch_size, seq_len, hidden_size]
            hidden_states = torch.mean(hidden_states, dim=0).to(dtype)
            outputs.last_hidden_state = (outputs.last_hidden_state + hidden_states) / 2
            del self.layer_out
            self.layer_out = ()
            torch.cuda.empty_cache()
            return outputs

    
    return GAModel