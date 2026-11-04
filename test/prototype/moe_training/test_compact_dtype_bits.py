# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao.prototype.moe_training.kernels import mxfp8
from torchao.utils import is_sm_at_least_100


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm_at_least_100()
    or not mxfp8._mxfp8_compact_cuda_available,
    reason="requires the compact CUDA extension on SM100+",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("columns", [128, 129])
@pytest.mark.parametrize("storage_offset", [0, 1])
def test_compact_pad_preserves_raw_dtype_bits(dtype, columns, storage_offset):
    tokens, experts = 8, 64
    active = [1, 3, 5, 7, 13, 24, 48, 63]
    counts = [int(i in active) for i in range(experts)]
    offsets = torch.tensor(counts, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    if dtype == torch.float32:
        integer_dtype = torch.int32
        patterns = [
            0,
            -2147483648,
            0x3F800000,
            -1082130432,
            0x7F800000,
            0x7FC00001,
            0x01234567,
            -1,
        ]
    else:
        integer_dtype = torch.int16
        patterns = [0, -32768, 0x3F80, -16512, 0x7F80, 0x7FC1, 0x0123, -1]
    numel = tokens * columns
    raw = torch.tensor(patterns, dtype=integer_dtype, device="cuda")
    raw = raw.repeat((numel + storage_offset + len(patterns) - 1) // len(patterns))
    x = raw[storage_offset : storage_offset + numel].view(dtype).view(tokens, columns)
    padded, starts, ends = mxfp8.fused_pad_token_groups_compact_cuda(x, offsets, 128)
    assert padded.shape == (1024, columns)
    starts_cpu = starts.cpu().tolist()
    for token, expert in enumerate(active):
        assert torch.equal(
            padded[starts_cpu[expert]].view(integer_dtype), x[token].view(integer_dtype)
        )
    restored = mxfp8.fused_unpad_token_groups_cuda(padded, offsets, starts, tokens, 128)
    assert torch.equal(restored.view(integer_dtype), x.view(integer_dtype))
