# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from typing import Any, Optional, List

import torch
import torch.nn as nn

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight

from .layer import SMTLayer, linearZExMod
import bitsandbytes as bnb

if is_bnb_available():
    class SparseLinear8bitLt(nn.Module, SMTLayer):
        def __init__(self, base_layer, adapter_name, index_list, block_size, **kwargs):
            super().__init__()
            SMTLayer.__init__(self, base_layer, index_list, block_size, **kwargs)
            self.adapter_name = adapter_name
            self.index_list = index_list
            self.block_size = block_size
            self.weight_backup = None
            assert len(self.index_list) > 0, "No indices provided for sparse layer"

            # Freezing the pre-trained weight matrix
            self.get_base_layer().weight.requires_grad = False
            self._active_adapter = adapter_name
            self.update_layer(self.adapter_name, self.block_size, self.index_list)

            # Apply specialized sparse linear multiplication function
            self.fn = linearZExMod.apply
            
        def get_delta_weight(self, adapter, original_weight):
            device = self.smt_weight[adapter].device
            dtype = self.smt_weight[adapter].dtype

            delta = torch.zeros_like(original_weight)

            for i, index in enumerate(self.index_list):
                smt_block = self.smt_weight[adapter].data[i * self.block_size:(i+1) * self.block_size].view(self.block_size, self.block_size)
                original_block = original_weight.data[index[0] * self.block_size:(index[0]+1) * self.block_size,
                                                            index[1] * self.block_size:(index[1]+1) * self.block_size]
                delta[index[0] * self.block_size:(index[0]+1) * self.block_size,
                    index[1] * self.block_size:(index[1]+1) * self.block_size] = smt_block - original_block

            return delta

        def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                return

            # Store (quantized) weight
            if self.weight_backup is None:
                self.weight_backup = self.get_base_layer().weight.clone()

            for active_adapter in adapter_names:
                if active_adapter not in self.smt_weight.keys():
                    continue

                warnings.warn(
                    "Merge SMT module to 8-bit linear may get different generations due to rounding errors."
                )

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                output = dequantize_bnb_weight(weight, state=state)
                smt_data = self.get_delta_weight(active_adapter, output)

                w_data = output.to(smt_data.dtype).to(smt_data.device) + smt_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                
                self.get_base_layer().weight = bnb.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
                self.merged_adapters.append(active_adapter)

                weight_diff = w_data.norm()
                # print(f"Weight difference after merge: {weight_diff}")
                assert weight_diff > 0, "Weights did not change after merge"

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            if self.weight_backup is None:
                warnings.warn("No weight backup found. Cannot perform accurate unmerge.")
                return

            base_layer = self.get_base_layer()

            # Restore original (quantized) weights
            base_layer.weight.data = self.weight_backup.clone()

            # Update SMT weights
            # I'm pretty sure this is unnecessary, so it's commented out
            # for active_adapter in self.merged_adapters:
            #     if active_adapter in self.smt_weight.keys():
            #         for i, index in enumerate(self.index_list):
            #             self.smt_weight[active_adapter].data[i * self.block_size: i * self.block_size + self.block_size, :] = \
            #                 base_layer.weight.data[index[0] * self.block_size: index[0] * self.block_size + self.block_size,
            #                     index[1] * self.block_size: index[1] * self.block_size + self.block_size]

            self.merged_adapters = []

            # Free up memory
            del self.weight_backup
            self.weight_backup = None
                    
        def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            self._check_forward_args(input, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.weight_backup is None:
                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                self.weight_backup = dequantize_bnb_weight(weight, state=state)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(input, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(input, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(input, *args, **kwargs)
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter in self.smt_weight.keys():
                        smt_weight = self.smt_weight[active_adapter]
                        result = self.fn(input, smt_weight, self.index_list, self.weight_backup, self.block_size)
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "smt." + rep

if is_bnb_4bit_available():
    class SparseLinear4bit(torch.nn.Module, SMTLayer):
        # SMT implemented in a dense layer
        def __init__(self, base_layer, adapter_name, index_list, block_size, **kwargs):
            super().__init__()
            SMTLayer.__init__(self, base_layer, index_list, block_size, **kwargs)
            self.adapter_name = adapter_name
            self.index_list = index_list
            self.block_size = block_size
            self.weight_backup = None
            assert len(self.index_list) > 0, "No indices provided for sparse layer"

            # Freezing the pre-trained weight matrix
            self.get_base_layer().weight.requires_grad = False
            self._active_adapter = adapter_name
            self.update_layer(self.adapter_name, self.block_size, self.index_list)

            # Apply specialized sparse linear multiplication function
            self.fn = linearZExMod.apply
            
        def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                return

            # Store (quantized) weight
            if self.weight_backup is None:
                self.weight_backup = self.get_base_layer().weight.clone()

            for active_adapter in adapter_names:
                if active_adapter not in self.smt_weight.keys():
                    continue

                warnings.warn(
                    "Merge SMT module to 4-bit linear may get different generations due to rounding errors."
                )

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__

                output = dequantize_bnb_weight(weight, state=weight.quant_state)
                smt_data = self.get_delta_weight(active_adapter, output)

                w_data = output.to(smt_data.dtype).to(smt_data.device) + smt_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                
                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)
                self.merged_adapters.append(active_adapter)

                weight_diff = w_data.norm()
                # print(f"Weight difference after merge: {weight_diff}")
                assert weight_diff > 0, "Weights did not change after merge"

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            if self.weight_backup is None:
                warnings.warn("No weight backup found. Cannot perform accurate unmerge.")
                return

            base_layer = self.get_base_layer()

            # Restore original (quantized) weights
            base_layer.weight.data = self.weight_backup.clone()

            # Update SMT weights
            # I'm pretty sure this is unnecessary, so it's commented out
            # for active_adapter in self.merged_adapters:
            #     if active_adapter in self.smt_weight.keys():
            #         for i, index in enumerate(self.index_list):
            #             self.smt_weight[active_adapter].data[i * self.block_size: i * self.block_size + self.block_size, :] = \
            #                 base_layer.weight.data[index[0] * self.block_size: index[0] * self.block_size + self.block_size,
            #                     index[1] * self.block_size: index[1] * self.block_size + self.block_size]

            self.merged_adapters = []

            # Free up memory
            del self.weight_backup
            self.weight_backup = None
                    
        def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            self._check_forward_args(input, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.weight_backup is None:
                weight = self.get_base_layer().weight
                self.weight_backup = dequantize_bnb_weight(weight, state=weight.quant_state)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(input, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(input, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(input, *args, **kwargs)
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter in self.smt_weight.keys():
                        smt_weight = self.smt_weight[active_adapter]
                        result = self.fn(input, smt_weight, self.index_list, self.weight_backup, self.block_size)
                        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                        # The reason is that in some cases, an error can occur that backprop
                        # does not work on a manipulated view. This issue may be solved with
                        # newer PyTorch versions but this would need extensive testing to be
                        # sure.
                        result = result.clone()
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "smt." + rep
