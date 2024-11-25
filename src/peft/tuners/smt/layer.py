import math
import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from .config import SMTConfig

class SMTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("smt_weight",)
    # All names of other parameters that may contain adapter-related parameters
    # other_param_names = ("index_list", "block_size")

    def __init__(self, base_layer: nn.Module, index_list: list, block_size: int, **kwargs):
        self.base_layer = base_layer
        self.index_list = index_list
        self.block_size = block_size
        self.smt_weight = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, block_size, index_list):
        with gather_params_ctx(self.get_base_layer().weight):
            base_layer = self.get_base_layer()
            orig_weight = base_layer.weight
            bnb_param_type = get_bnb_param_type(orig_weight)
            dtype = orig_weight.dtype

            if bnb_param_type:
                weight_tensor = dequantize_module_weight(base_layer)
            elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
                weight_tensor = orig_weight
            else:
                raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")
            
            # maintain a new trainable parameter
            weight = torch.empty(len(index_list) * block_size, block_size,
                dtype=weight_tensor.dtype, device=weight_tensor.device)

            # project original weight parameters to new trainable parameters 'smt_weight'
            for i, index in enumerate(index_list):
                weight.data[i * block_size: i * block_size + block_size, :] = \
                    weight_tensor.data[index[0] * block_size: index[0] * block_size + block_size, \
                                index[1] * block_size: index[1] * block_size + block_size]
            self.smt_weight[adapter_name] = nn.Parameter(weight)
            # print(weight_tensor.shape, weight.shape, len(index_list))
            self.index_list = index_list
            self._move_adapter_to_device_of_base_layer(adapter_name)
            self.set_adapter(self.active_adapters)

    def reset_smt_parameters(self, adapter_name):
        if adapter_name in self.smt_weight.keys():
            # Not sure what to initialize this to. zero or default nn.Linear?
            # nn.init.normal_(self.smt_weight[adapter_name], mean=0.0, std=0.02)
            nn.init.zeros_like(self.smt_weight[adapter_name])

    def _check_forward_args(self, input, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(input) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(input)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def _mixed_batch_forward(
        self, input: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(input, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.smt_weight.keys():
                continue

            smt_weight = self.smt_weight[active_adapter]

            # getting the sub-batch
            sub_batch = input[sub_batch_indices_list[i]]

            # Apply SMT transformation
            smt_output = self.fn(sub_batch, smt_weight, self.index_list, self.base_layer.weight)

            # Update the corresponding indices of the linear layer output
            result[sub_batch_indices_list[i]] = smt_output.to(torch_result_dtype)
            
        return result

class SparseLinear(nn.Module, SMTLayer):
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

    def get_delta_weight(self, adapter):
        device = self.smt_weight[adapter].device
        dtype = self.smt_weight[adapter].dtype

        delta = torch.zeros_like(self.base_layer.weight)

        for i, index in enumerate(self.index_list):
            smt_block = self.smt_weight[adapter].data[i * self.block_size:(i+1) * self.block_size].view(self.block_size, self.block_size)
            original_block = self.base_layer.weight.data[index[0] * self.block_size:(index[0]+1) * self.block_size,
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

        if self.weight_backup is None:
            self.weight_backup = self.get_base_layer().weight.clone()

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.smt_weight.keys():
                delta_weight = self.get_delta_weight(active_adapter)

                if safe_merge:
                    new_weight = base_layer.weight.data + delta_weight
                    if not torch.isfinite(new_weight).all():
                        raise ValueError(f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken")
                    base_layer.weight.data = new_weight
                else:
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

        weight_diff = delta_weight.norm()
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
        orig_dtype = base_layer.weight.data.dtype

        # Restore original weights
        base_layer.weight.data = self.weight_backup.clone().to(orig_dtype)

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
        device_dtype = input.dtype
        
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
                    base_weight = self.base_layer.weight.to(device_dtype)
                    # for i, index in enumerate(self.index_list):
                    #     self.base_layer.weight.data[index[0] * self.block_size: index[0] * self.block_size + self.block_size,
                    #                     index[1] * self.block_size: index[1] * self.block_size + self.block_size] = \
                    #         smt_weight.data[i * self.block_size: i * self.block_size + self.block_size, :]
                    result = self.fn(input, smt_weight, self.index_list, base_weight, self.block_size)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "smt." + rep

class linearZExMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selected_weight, matrix_index_list, weight, block_size):
        device_dtype = input.dtype
        weight = weight.to(device_dtype)
        selected_weight = selected_weight.to(device_dtype)

        # Maintain partial input in `input_list`
        input_list = []
        for index in matrix_index_list:
            input_list.append(input[:, :, index[1] * block_size: (index[1] + 1) * block_size])

        # Create a copy of the weight tensor
        updated_weight = weight.clone()

        # Update the selected blocks in the weight copy
        for i, index in enumerate(matrix_index_list):
            updated_weight[index[0] * block_size:(index[0] + 1) * block_size,
                           index[1] * block_size:(index[1] + 1) * block_size] = \
                selected_weight[i * block_size:(i + 1) * block_size, :].view(block_size, block_size)

        # Save for backward
        ctx.save_for_backward(input, selected_weight, weight)
        ctx.matrix_index_list = matrix_index_list
        ctx.block_size = block_size

        # Compute output using the updated weight
        output = torch.matmul(input, updated_weight.t())

        return output.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        device_dtype = grad_output.dtype
        input, selected_weight, weight = (t.to(device_dtype) for t in ctx.saved_tensors)
        matrix_index_list = ctx.matrix_index_list
        block_size = ctx.block_size

        # Calculate gradient for input
        grad_input = torch.matmul(grad_output, weight)

        # Calculate gradient for selected_weight
        grad_selected_weight = torch.zeros_like(selected_weight)
        for i, index in enumerate(matrix_index_list):
            input_block = input[:, :, index[1] * block_size:(index[1] + 1) * block_size]
            grad_output_block = grad_output[:, :, index[0] * block_size:(index[0] + 1) * block_size]
            grad_selected_weight[i * block_size: (i + 1) * block_size, :] = torch.sum(
                torch.matmul(grad_output_block.permute(0, 2, 1), input_block),
                dim=0
            ).view(-1, block_size)

        # We don't compute gradients for matrix_index_list, weight, or block_size
        return grad_input, grad_selected_weight, None, None, None