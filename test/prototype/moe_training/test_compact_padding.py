# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_sm_at_least_100

if not torch.cuda.is_available() or not is_sm_at_least_100():
    pytest.skip("requires SM100+", allow_module_level=True)

from torchao.prototype.moe_training.kernels import mxfp8
from torchao.prototype.moe_training.utils import pad_token_groups


@pytest.mark.skipif(
    not mxfp8._mxfp8_compact_cuda_available, reason="requires compact CUDA extension"
)
@pytest.mark.parametrize(
    "tokens,compact_rows", [(0, 8192), (8, 1024), (64, 8192), (128, 8192)]
)
def test_compact_and_legacy_padding_contracts(tokens, compact_rows):
    groups = [1] * min(tokens, 64) + [0] * max(0, 64 - tokens)
    if tokens > 64:
        groups[-1] += tokens - 64
    offsets = torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    x = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda")
    legacy, legacy_starts, legacy_ends = mxfp8.fused_pad_token_groups_cuda(
        x, offsets, 128
    )
    compact, starts, ends = mxfp8.fused_pad_token_groups_compact_cuda(x, offsets, 128)
    selected, selected_starts, selected_ends = pad_token_groups(
        x, offsets, alignment_size=128, compact=True
    )
    legacy_rows = ((tokens + 64 * 128 + 127) // 128) * 128
    assert legacy.shape == (legacy_rows, 128)
    assert compact.shape == selected.shape == (compact_rows, 128)
    torch.testing.assert_close(compact, legacy[:compact_rows], rtol=0, atol=0)
    torch.testing.assert_close(starts, legacy_starts, rtol=0, atol=0)
    torch.testing.assert_close(ends, legacy_ends, rtol=0, atol=0)
    torch.testing.assert_close(selected, compact, rtol=0, atol=0)
    torch.testing.assert_close(selected_starts, starts, rtol=0, atol=0)
    torch.testing.assert_close(selected_ends, ends, rtol=0, atol=0)


@pytest.mark.skipif(
    not mxfp8._mxfp8_cuda_kernels_available, reason="requires native CUDA extension"
)
def test_compact_request_falls_back_when_kernel_missing(monkeypatch):
    monkeypatch.setattr(mxfp8, "_mxfp8_compact_cuda_available", False)
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor([1] * 8 + [0] * 56, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    actual = pad_token_groups(x, offsets, alignment_size=128, compact=True)
    expected = mxfp8.fused_pad_token_groups_cuda(x, offsets, 128)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(
    not mxfp8._mxfp8_compact_cuda_available, reason="requires compact CUDA extension"
)
def test_compact_padding_opcheck():
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor([1] * 8 + [0] * 56, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    torch.library.opcheck(
        torch.ops.torchao.fused_pad_token_groups_compact.default, (x, offsets, 128)
    )
