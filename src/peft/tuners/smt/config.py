from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Dict

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SMTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SMTModel`].

    Args:
        sparsity_ratio (`float`):
            The sparsity rate for SMT. This determines what proportion of parameters will be updated.
        sparse_modules (`Union[List[str], str]`):
            The names of the modules to apply SMT to. If this is specified, only the modules with the specified
            names will be replaced with SMT layers.
        block_size (`int`):
            The l x l dimensions of the submatrix blocks.
        selection_method (`str`):
            Which selection method to use. Available options: GW, AW, MW.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply SMT to. This is an alias for `sparse_modules` and will be used if
            `sparse_modules` is not specified.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from SMT layers to be set as trainable and saved in the final checkpoint.
    """

    sparsity_ratio: float = field(
        default=0.01,
        metadata={"help": "The sparsity rate for SMT."},
    )
    sparse_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with SMT."
        },
    )
    block_size: int = field(
        default=256,
        metadata={"help": "The l x l dimensions of the submatrix blocks."},
    )
    selection_method: str = field(
        default="GW",
        metadata={"help": "Which selection method to use. Available options: GW, AW, MW."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "Alias for sparse_modules. List of module names or regex expression of the module names to replace with SMT."
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from SMT layers to be set as trainable and saved in the final checkpoint."
        },
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "The number of warmup steps to use for the selection method."},
    )
    dataloader: Optional[Any] = field(
        default=None,
        metadata={"help": "The dataloader to use for the warmup steps. Required for GW and AWQ selection methods."},
    )
    # DO. NOT. PASS. THIS.
    # TODO: Find elsewhere to store selected_indices.
    # Perhaps in the module's state_dict? I don't have much experience with altering it.
    selected_indices: Optional[Dict[str, List[tuple]]] = field(
        default=None,
        metadata={"help": "Selected indices for each sparse module. This is set internally and should not be set by users."}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from SMT layers to be set as trainable and saved in the final checkpoint."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.SMT
        if self.sparse_modules is None and self.target_modules is not None:
            self.sparse_modules = self.target_modules
        self.sparse_modules = (
            set(self.sparse_modules) if isinstance(self.sparse_modules, list) else self.sparse_modules
        )

        if self.sparsity_ratio <= 0 or self.sparsity_ratio > 1:
            raise ValueError("sparsity_ratio must be between 0 and 1")

        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")

        if self.sparse_modules is None:
            raise ValueError("Either sparse_modules or target_modules must be specified")

        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be at least 1")

        if not self.inference_mode and self.selection_method in ["GW", "AW"] and self.dataloader is None:
            raise ValueError("dataloader must be provided for GW and AWQ selection methods")