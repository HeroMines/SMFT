import torch
import torch.nn as nn

class LinearLayer_MatrixSparsity(torch.nn.Module):
    def __init__(self, weight, bias=None, index_list=[], block_size=256):
        super(LinearLayer_MatrixSparsity, self).__init__()
        self.weight = weight
        # freeze all the weight and not passing full weight into optimizer
        self.weight.requires_grad = False
        self.bias = bias
        self.index_list = index_list
        self.block_size = block_size
        assert len(index_list) > 0, "No indices provided for sparse layer"

        # maintain a new trainable parameter 'selected_weight'
        self.selected_weight = torch.empty(len(index_list) * self.block_size, self.block_size,
                                      dtype=self.weight.dtype, device=self.weight.device)
        self.selected_weight.requires_grad = True
        
        # project original weight parameters to new trainable parameters 'selected_weight'
        for i, index in enumerate(self.index_list):
            self.selected_weight.data[i * self.block_size: i * self.block_size + self.block_size, :] = \
                self.weight.data[index[0] * self.block_size: index[0] * self.block_size + self.block_size, \
                               index[1] * self.block_size: index[1] * self.block_size + self.block_size]
        self.selected_weight = nn.Parameter(self.selected_weight)
        
        # apply a specialized sparse linear multiplication function
        self.fn = linearZ.apply

        # register backward hook
        # self.selected_weight.register_hook(self.update_original_weight)

        # print(f"Initialized LinearLayer_MatrixSparsity with {len(index_list)} sub-matrices")

    def forward(self, input):
        for i, index in enumerate(self.index_list):
            self.weight.data[index[0] * self.block_size: index[0] * self.block_size + self.block_size,
                              index[1] * self.block_size: index[1] * self.block_size + self.block_size] = \
                 self.selected_weight.data[i * self.block_size: i * self.block_size + self.block_size, :]
        
        return self.fn(input, self.selected_weight, self.index_list, self.weight, self.block_size)

    # def update_original_weight(self, grad):
    #     with torch.no_grad():
    #         for i, index in enumerate(self.index_list):
    #             self.weight.data[index[0] * self.block_size: index[0] * self.block_size + self.block_size,
    #                              index[1] * self.block_size: index[1] * self.block_size + self.block_size] = \
    #                 self.selected_weight.data[i * self.block_size: i * self.block_size + self.block_size, :]
    #     return grad

class linearZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selected_weight, matrix_index_list, weight, block_size):
        # maintain parital input in `input_list`
        input_list = []
        for index in matrix_index_list:
            input_list.append(input[:, :, index[1] * block_size: index[1] * block_size + block_size])

        # save the partial input(`input_list`) and sub-matrices index(`matrix_index_list`) for backward propagation
        ctx.list1 = input_list
        ctx.list2 = matrix_index_list
        ctx.block_size = block_size
        
        ctx.save_for_backward(weight)
        output = torch.matmul(input, weight.t())

        # memory free
        del weight
        del input_list
        del matrix_index_list
        del block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # fetch weight, partial input('input_list'), and sub-matrices index('matrix_index_list')
        weight, = ctx.saved_tensors
        input_list = ctx.list1
        matrix_index_list = ctx.list2
        block_size = ctx.block_size
        
        # calculate the partial gradient and maintain it in 'grad_weight'
        grad_weight = torch.empty(len(input_list) * block_size, block_size)
        for i in range(len(input_list)):
            index = matrix_index_list[i]
            grad_weight[i * block_size: i * block_size + block_size, :] = torch.sum(torch.matmul(grad_output.permute(0, 2, 1) \
                [:, index[0] * block_size: index[0] * block_size + block_size, :], input_list[i]), dim=0)
        
        # calculate dl/dz for backward propagation
        grad_input = torch.matmul(grad_output, weight)
        # return gradient for activation and selected sub-matrices

        # memory free
        del weight
        del input_list
        del matrix_index_list
        del block_size

        return grad_input, grad_weight.to(grad_output.device), None, None, None

class SMT():
    def __init__(self, model, dataloader=None, warmup_steps=100, sparsity_ratio=0.01, sparse_modules=['q_proj', 'k_proj', 'v_proj'], block_size=256, selection_method="GW"):
        self.model = model
        self.dataloader = dataloader
        self.warmup_steps = warmup_steps
        self.sparsity_ratio = sparsity_ratio
        self.gradient_sum = {}
        self.activation_magnitudes = {}
        self.selected_indices = {}
        self.trainable_params = list()
        self.sparse_modules = sparse_modules
        self.block_size = block_size
        self.selection_method = selection_method if dataloader is not None else "MW"
        assert self.selection_method in ["AW", "GW", "MW"], "Invalid sparse sub-matrix selection method"

        self.select_submatrices()
        print(f"Selected {sum(len(indices) for indices in self.selected_indices.values())} sub-matrices in total.")
        print(f"Warmup complete. Selected {len(self.selected_indices)} layers for sparse tuning.")
        self.replace_layers()

    def select_submatrices(self):
        if self.selection_method == 'AW':
            # Not sure if this implemented correctly
            return self.compute_activation_magnitudes(self.dataloader)
        elif self.selection_method == "MW":
            # Magnitude-based Weight selection
            # Selects the blocks with the highest average magnitude
            # Not in the paper, just something I thought of while looking at SIFT (Song et. al)
            # Seems comparable to AWQ without having to gather activations
            return self.mw_select_submatrices()
        else:
            # GW is the best selection method
            return self.warmup(self.dataloader)

    def warmup(self, dataloader):
        self.model.train()
        for step, batch in enumerate(dataloader):
            if step >= self.warmup_steps:
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
                if any(proj in name for proj in self.sparse_modules):
                    if param.grad is not None:
                        if name not in self.gradient_sum:
                            self.gradient_sum[name] = torch.zeros_like(param.grad)
                        self.gradient_sum[name] += param.grad.abs()

            # Zero gradients for next iteration
            self.model.zero_grad()

        # print(f"Gradient sum keys: {self.gradient_sum.keys()}")
        self.gw_select_submatrices()

    def gw_select_submatrices(self):
        all_block_averages = []
        all_indices = []
        
        for name, grad_sum in self.gradient_sum.items():
            out_blocks, in_blocks = (grad_sum.shape[0] // self.block_size, grad_sum.shape[1] // self.block_size)

            # Reshape to (out_blocks, block_size, in_blocks, block_size)
            # Ex: (512, 2048) -> (2, 256, 8, 256)
            reshaped = grad_sum.view(out_blocks, self.block_size, in_blocks, self.block_size)

            # Average the absolute values within each sub-matrix
            block_averages = reshaped.mean(dim=(1,3)).abs()

            all_block_averages.append(block_averages.view(-1))
            all_indices.extend([(name, i // in_blocks, i % in_blocks) for i in range(out_blocks * in_blocks)])

        # Concatenate all block averages
        all_block_averages = torch.cat(all_block_averages)

        # Select top Y% across all Q, K, V layers
        total_blocks = len(all_block_averages)
        num_select = int(self.sparsity_ratio * total_blocks)
        _, top_indices = torch.topk(all_block_averages, num_select)

        # Save selected indices
        for idx in top_indices:
            name, out_idx, in_idx = all_indices[idx]
            if name not in self.selected_indices:
                self.selected_indices[name] = []
            self.selected_indices[name].append((out_idx, in_idx))

            # print(f"Selected {len(self.selected_indices[name])} indices for {name}")

    def compute_activation_magnitudes(self, dataloader):
        self.model.eval()
        for step, batch in enumerate(dataloader):
            if step >= self.warmup_steps:
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
                    if isinstance(module, nn.Linear) and any(proj in name for proj in self.sparse_modules):
                        if name not in self.activation_magnitudes:
                            self.activation_magnitudes[name] = []
                        self.activation_magnitudes[name].append(module.weight.abs().mean(dim=1))

        for name in self.activation_magnitudes:
            self.activation_magnitudes[name] = torch.stack(self.activation_magnitudes[name]).mean(dim=0)
        
        self.aw_select_submatrices()

    def aw_select_submatrices(self):
        all_magnitudes = []
        all_indices = []

        for name, magnitudes in self.activation_magnitudes.items():
            # Calculate the number of blocks in output and input dimensions
            out_blocks, in_blocks = (magnitudes.shape[0] // self.block_size, self.model.get_submodule(name).weight.shape[1] // self.block_size)

            # Average the magnitudes for each output block
            block_magnitudes = magnitudes.view(out_blocks, self.block_size).mean(dim=1)
            all_magnitudes.append(block_magnitudes)

            # Generate all possible (out_idx, in_idx) combinations for this layer
            all_indices.extend([(name, i, j) for i in range(out_blocks) for j in range(in_blocks)])

        # Concatenate all magnitudes and repeat each in_blocks times
        # This is because each output block magnitude applies to all its input blocks
        all_magnitudes = torch.cat([m.repeat(in_blocks) for m in all_magnitudes])

        # Select the top num_select blocks based on magnitudes
        num_select = int(self.sparsity_ratio * len(all_magnitudes))
        _, top_indices = torch.topk(all_magnitudes, num_select)

        # Save selected indices
        for idx in top_indices:
            name, out_idx, in_idx = all_indices[idx]
            if name not in self.selected_indices:
                self.selected_indices[name] = []
            self.selected_indices[name].append((out_idx, in_idx))

    def mw_select_submatrices(self):
        all_block_magnitudes = []
        all_indices = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(proj in name for proj in self.sparse_modules):
                weight = module.weight.data
                out_blocks, in_blocks = (weight.shape[0] // self.block_size, weight.shape[1] // self.block_size)

                # Reshape weight to blocks
                weight_blocks = weight.view(out_blocks, self.block_size, in_blocks, self.block_size)
                
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
        num_select = int(self.sparsity_ratio * total_blocks)
        _, top_indices = torch.topk(all_block_magnitudes, num_select)

        # Save selected indices
        for idx in top_indices:
            name, out_idx, in_idx = all_indices[idx]
            if name not in self.selected_indices:
                self.selected_indices[name] = []
            self.selected_indices[name].append((out_idx, in_idx))

    def replace_layers(self):
        layers_to_replace = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = name+'.weight' if self.selection_method == "GW" else name
                if layer_name in self.selected_indices and any(proj in name for proj in self.sparse_modules):
                    layers_to_replace.append((name, module))
                # Freeze layers that don't have selected sub-matrices
                else:
                    module.weight.requires_grad = False

        for name, module in layers_to_replace:
            # print(f"Replacing layer {name} with LinearLayer_MatrixSparsity")
            layer_name = name+'.weight' if self.selection_method == "GW" else name
            new_layer = LinearLayer_MatrixSparsity(module.weight, module.bias, self.selected_indices[layer_name], self.block_size)
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = self.model.get_submodule(parent_name)
            setattr(parent_module, child_name, new_layer)
            self.trainable_params.extend(new_layer.parameters())

        # print(f"Replaced {len(layers_to_replace)} layers in total")

    def get_trainable_params(self):
        return iter(p for p in self.trainable_params if p.requires_grad)

    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        percentage = (trainable_params / total_params) * 100

        print(
            f"trainable params: {trainable_params:,d} || all params: {total_params:,d} || trainable%: {percentage:.2f}"
        )