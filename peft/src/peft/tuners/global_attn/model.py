import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)

from .config import GAConfig
from .layer import get_ga_model


class GAPEFTModel(BaseTuner):
    preftx: str = "ga_"
    class_type = None
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
    
    def _check_new_adapter_config(self, config: GAConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )
    
    def _check_target_module_exists(self, ga_config, key):
        if key == "model":
            return True
        return False
    
    def _create_and_replace(
        self,
        ga_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        layerid=None,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        new_module = self._create_new_module(ga_config, adapter_name, target)
        if adapter_name != self.active_adapter:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # print(type(parent), type(new_module), type(child))
        new_module.load_state_dict(child.state_dict(), strict=False)
        device, dtype = child.device, child.dtype
        del child
        torch.cuda.empty_cache()
        new_module = new_module.to(device)
        new_module = new_module.to(dtype)
        
    def _mark_only_adapters_as_trainable(self, model) -> None:
        for n, p in model.named_parameters():
            if "global_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
    @staticmethod
    def _create_new_module(ga_config, adapter_name, target, **kwargs):
        #[TODO] ga_config is in fact useless
        new_module = get_ga_model(target.config)(
            target.config, base_model=target, adapter_name=adapter_name, **kwargs
        )
        GAPEFTModel.class_type = new_module.__class__
        return new_module
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, GAPEFTModel.class_type):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        return peft_config