from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils import (
    ModulesToSaveWrapper,
    _get_submodules,
)

from .layer import SMTLayer, SparseLinear
import bitsandbytes as bnb

def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs

class SMTModel(BaseTuner):
    prefix: str = "smt_"
    
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _prepare_adapter_config(self, config, model_config):
        if config.target_modules is None:
            config.target_modules = config.sparse_modules
        # TODO: Find a better place to put this
        # Putting it here is bad and I feel bad
        if not config.inference_mode:
            self.warmup_steps = config.warmup_steps
            self.sparsity_ratio = config.sparsity_ratio
            self.sparse_modules = config.sparse_modules
            self.block_size = config.block_size
            self.dataloader = config.dataloader
            self.gradient_sum = {}
            self.activation_magnitudes = {}
            self.selected_indices = {}
            self.selection_method = config.selection_method if self.dataloader is not None else "MW"
            self.select_submatrices(config)

            # Do not override this please
            # if hasattr(config, "exclude_modules") == False:
            #     config.exclude_modules = []
            # for name, module in self.model.named_modules():
            #     if isinstance(module, nn.Linear) and name not in self.selected_indices:
            #         config.exclude_modules.append(name)
            # # print(config.exclude_modules)
            # assert len(config.exclude_modules) > 0
            
            # remove dataloader after use
            config.dataloader = None

            # save the indices to config for later.
            config.selected_indices = self.selected_indices
        if hasattr(config, "exclude_modules") == False:
            config.exclude_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name not in config.selected_indices:
                config.exclude_modules.append(name)
        # print(config.exclude_modules)
        assert len(config.exclude_modules) > 0
        return config

    def _create_and_replace(
        self, 
        config, 
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # print(adapter_name, target_name, current_key, self.selected_indices[current_key])
        selected_weight_indices = config.selected_indices[current_key]
        
        kwargs = {
            "index_list": selected_weight_indices,
            "block_size": config.block_size,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        if isinstance(target, SMTLayer):
            target.update_layer(
                adapter_name,
                config.block_size,
                selected_weight_indices
            )
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

        return new_module

    @staticmethod
    def _check_target_module_exists(smt_config, key):
        return check_target_module_exists(smt_config, key)

    @staticmethod
    def _create_new_module(smt_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb
            
            from .bnb import SparseLinear8bitLt

        if is_bnb_4bit_available():
            from .bnb import SparseLinear4bit

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = SparseLinear8bitLt(target, adapter_name, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = SparseLinear4bit(target, adapter_name, **fourbit_kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            new_module = SparseLinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module
    
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model):
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
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
            if isinstance(module, SMTLayer):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        # Check that users only passed actually existing adapters.
        # Note: We cannot do this on the layer level, as each individual layer may not have each adapter. Still, we want
        # to check that there is at least one layer with the given name, or else something like typos can easily slip.
        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, SMTLayer):
                expected_adapters |= layer.smt_weight.keys()
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, SMTLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def set_adapter(self, adapter_name: str | list[str]):
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, SMTLayer):
                if module.merged:
                   warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                   module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def warmup(self, dataloader, config):
        self.model.train()
        for step, batch in enumerate(dataloader):
            if step >= config.warmup_steps:
                break
            # Move batch to the same device as the model
            batch = {k: v.to(self.model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Only pass input_ids, attention_mask, and labels (if present)
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }

            if 'labels' not in batch:
                # Create labels by shifting input_ids right by one position
                labels = inputs['input_ids'].clone()
                labels[:, :-1] = inputs['input_ids'][:, 1:]
                labels[:, -1] = -100  # -100 is typically used to ignore this position in loss calculation             
                inputs['labels'] = labels
            else:
                inputs['labels'] = batch['labels']

            # Forward pass
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            for name, param in self.model.named_parameters():
                layer_name = name.replace(".weight", "")
                if any(proj in layer_name for proj in config.target_modules):
                    if param.grad is not None:
                        if layer_name not in self.gradient_sum:
                            self.gradient_sum[layer_name] = torch.zeros_like(param.grad)
                        self.gradient_sum[layer_name] += param.grad.abs()

            # Zero gradients for next iteration
            self.model.zero_grad()

        # print(f"Gradient sum keys: {self.gradient_sum.keys()}")
        self.gw_select_submatrices(config)

    def gw_select_submatrices(self, config):
        all_block_averages = []
        all_indices = []
        
        for name, grad_sum in self.gradient_sum.items():
            out_blocks, in_blocks = (grad_sum.shape[0] // config.block_size, grad_sum.shape[1] // config.block_size)

            # Reshape to (out_blocks, block_size, in_blocks, block_size)
            # Ex: (512, 2048) -> (2, 256, 8, 256)
            reshaped = grad_sum.view(out_blocks, config.block_size, in_blocks, config.block_size)

            # Average the absolute values within each sub-matrix
            block_averages = reshaped.mean(dim=(1,3)).abs()

            all_block_averages.append(block_averages.view(-1))
            all_indices.extend([(name, i // in_blocks, i % in_blocks) for i in range(out_blocks * in_blocks)])

        # Concatenate all block averages
        all_block_averages = torch.cat(all_block_averages)

        # Select top Y% across all Q, K, V layers
        total_blocks = len(all_block_averages)
        num_select = int(config.sparsity_ratio * total_blocks)
        _, top_indices = torch.topk(all_block_averages, num_select)

        # Save selected indices
        for idx in top_indices:
            name, out_idx, in_idx = all_indices[idx]
            if name not in self.selected_indices:
                self.selected_indices[name] = []
            self.selected_indices[name].append((out_idx, in_idx))

            # print(f"Selected {len(self.selected_indices[name])} indices for {name}")

    def compute_activation_magnitudes(self, dataloader, config):
        self.model.eval()
        for step, batch in enumerate(dataloader):
            if step >= config.warmup_steps:
                break
            with torch.no_grad():
                # Move batch to the same device as the model
                batch = {k: v.to(self.model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                # Only pass input_ids, attention_mask
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }

                # Forward pass
                outputs = self.model(**inputs)

                for name, module in self.model.named_modules():
                    layer_name = name.replace(".weight", "")
                    if isinstance(module, nn.Linear) and any(proj in layer_name for proj in config.target_modules):
                        if name not in self.activation_magnitudes:
                            self.activation_magnitudes[layer_name] = []
                        self.activation_magnitudes[layer_name].append(module.weight.abs().mean(dim=1))

        for name in self.activation_magnitudes:
            self.activation_magnitudes[name] = torch.stack(self.activation_magnitudes[name]).mean(dim=0)
        
        self.aw_select_submatrices(config)

    def aw_select_submatrices(self, config):
        all_magnitudes = []
        all_indices = []

        for name, magnitudes in self.activation_magnitudes.items():
            # Calculate the number of blocks in output and input dimensions
            out_blocks, in_blocks = (magnitudes.shape[0] // config.block_size, self.model.get_submodule(name).weight.shape[1] // config.block_size)

            # Average the magnitudes for each output block
            block_magnitudes = magnitudes.view(out_blocks, config.block_size).mean(dim=1)
            all_magnitudes.append(block_magnitudes)

            # Generate all possible (out_idx, in_idx) combinations for this layer
            all_indices.extend([(name, i, j) for i in range(out_blocks) for j in range(in_blocks)])

        # Concatenate all magnitudes and repeat each in_blocks times
        # This is because each output block magnitude applies to all its input blocks
        all_magnitudes = torch.cat([m.repeat(in_blocks) for m in all_magnitudes])

        # Select the top num_select blocks based on magnitudes
        num_select = int(config.sparsity_ratio * len(all_magnitudes))
        _, top_indices = torch.topk(all_magnitudes, num_select)

        # Save selected indices
        for idx in top_indices:
            name, out_idx, in_idx = all_indices[idx]
            if name not in self.selected_indices:
                self.selected_indices[name] = []
            self.selected_indices[name].append((out_idx, in_idx))

    def mw_select_submatrices(self, config):
        all_block_magnitudes = []
        all_indices = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(proj in name for proj in config.target_modules):
                weight = module.weight.data
                if isinstance(module, bnb.nn.Linear8bitLt):
                    mod_weight = module.weight
                    state = module.state
                    if state.SCB is None:
                        state.SCB = weight.SCB
                    weight = dequantize_bnb_weight(mod_weight, state=state)
                if isinstance(module, bnb.nn.Linear4bit):
                    weight = dequantize_bnb_weight(module.weight, state=module.weight.quant_state)
                out_blocks, in_blocks = (weight.shape[0] // config.block_size, weight.shape[1] // config.block_size)

                # Reshape weight to blocks
                weight_blocks = weight.view(out_blocks, config.block_size, in_blocks, config.block_size)
                
                # Calculate magnitude of each block
                block_magnitudes = weight_blocks.mean(dim=(1,3)).abs()

                # Flatten the block magnitudes
                flat_magnitudes = block_magnitudes.view(-1)

                all_block_magnitudes.append(flat_magnitudes)
                all_indices.extend([(name, i // in_blocks, i % in_blocks) for i in range(out_blocks * in_blocks)])

        # Concatenate all block magnitudes
        all_block_magnitudes = torch.cat(all_block_magnitudes)

        # Select top Y% across all Q, K, V layers
        total_blocks = len(all_block_magnitudes)
        num_select = int(config.sparsity_ratio * total_blocks)
        _, top_indices = torch.topk(all_block_magnitudes, num_select)

        # Save selected indices
        for idx in top_indices:
            name, out_idx, in_idx = all_indices[idx]
            if name not in self.selected_indices:
                self.selected_indices[name] = []
            self.selected_indices[name].append((out_idx, in_idx))

    def select_submatrices(self, config):
        if self.selection_method == 'AW':
            # Not sure if this implemented correctly
            return self.compute_activation_magnitudes(config.dataloader, config)
        elif self.selection_method == "MW":
            # Magnitude-based Weight selection
            # Selects the blocks with the highest average magnitude
            # Not in the paper, just something I thought of while looking at SIFT (Song et. al)
            # Seems comparable to AW without having to gather activations
            return self.mw_select_submatrices(config)
        else:
            # GW is the best selection method
            return self.warmup(config.dataloader, config)

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, SMTLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the SMT layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)