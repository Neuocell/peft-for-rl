# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Compatibility utilities for different versions of transformers library.
"""

import importlib.metadata
from functools import lru_cache
from typing import Optional

from packaging import version


def patch_peft_tp_adapter_load_for_data_parallel() -> None:
    """Avoid PEFT's TP-only import when loading LoRA under plain FSDP.

    PEFT 0.19 calls its tensor-parallel state-dict helper whenever torch.distributed
    is initialized.  That helper imports APIs absent from Transformers 4.57.3,
    even when no layer has a tensor-parallel device mesh.  Preserve the original
    helper for real HF tensor-parallel models and skip it only for data-parallel
    models, where it would be a no-op after the import.
    """
    from peft.utils import save_and_load

    current = save_and_load._maybe_shard_state_dict_for_tp
    if getattr(current, "_verl_data_parallel_guard", False):
        return

    def guarded(model, state_dict, adapter_name):
        for module in model.modules():
            get_base_layer = getattr(module, "get_base_layer", None)
            if not callable(get_base_layer):
                continue
            base_layer = get_base_layer()
            if (
                getattr(base_layer, "_hf_tp_plan", None) is not None
                and getattr(base_layer, "_hf_device_mesh", None) is not None
            ):
                return current(model, state_dict, adapter_name)
        return None

    guarded._verl_data_parallel_guard = True
    save_and_load._maybe_shard_state_dict_for_tp = guarded

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check
