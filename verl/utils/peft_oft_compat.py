"""Compatibility patches for PEFT OFT under FSDP/checkpointed RL training."""

from __future__ import annotations

import torch


def patch_peft_oft_no_inplace_skew() -> None:
    """Avoid PEFT OFT in-place indexed writes that break checkpoint backward.

    PEFT 0.19.1 builds skew-symmetric matrices with
    ``matrix[:, rows, cols] = vec``. With PyTorch checkpointing/FSDP this can
    raise a view/in-place autograd error. The scatter form below is
    mathematically equivalent but out-of-place.
    """

    try:
        from peft.tuners.oft.layer import OFTRotationModule
    except Exception:
        return

    if getattr(OFTRotationModule, "_verl_no_inplace_skew_patch", False):
        return

    def _pytorch_skew_symmetric(self, vec: torch.Tensor, block_size: int) -> torch.Tensor:
        batch_size = vec.shape[0]
        flat_size = block_size * block_size
        flat = vec.new_zeros(batch_size, flat_size)
        upper_idx = (self.rows * block_size + self.cols).to(device=vec.device)
        upper_idx = upper_idx.unsqueeze(0).expand(batch_size, -1)
        flat = flat.scatter(1, upper_idx, vec)
        matrix = flat.reshape(batch_size, block_size, block_size)
        return matrix - matrix.transpose(-2, -1)

    OFTRotationModule._pytorch_skew_symmetric = _pytorch_skew_symmetric
    OFTRotationModule._verl_no_inplace_skew_patch = True
