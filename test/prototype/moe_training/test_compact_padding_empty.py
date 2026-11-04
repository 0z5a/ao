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

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels import mxfp8
from torchao.prototype.moe_training.nvfp4_training import nvfp4_grouped_mm
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


@pytest.mark.skipif(
    not mxfp8._mxfp8_compact_cuda_available, reason="requires compact CUDA extension"
)
@pytest.mark.parametrize(
    "kernel_preference",
    [
        KernelPreference.AUTO,
        KernelPreference.TRITON,
        pytest.param(
            KernelPreference.CUTEDSL,
            marks=pytest.mark.skipif(
                not cutedsl_nvfp4_kernels_available(), reason="requires CuteDSL"
            ),
        ),
    ],
)
@pytest.mark.parametrize("first_active", [0, 28, 56])
def test_sparse_compact_padding_with_device_asserts(
    monkeypatch, kernel_preference, first_active
):
    monkeypatch.setattr(nvfp4_grouped_mm, "_DEVICE_ASSERTS", True)
    torch.manual_seed(123)
    counts = [int(first_active <= i < first_active + 8) for i in range(64)]
    offsets = torch.tensor(counts, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.randn(
        64, 256, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    dy = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = torch.cat(
        [x_ref[i : i + 1] @ weight_ref[first_active + i].T for i in range(8)]
    )
    actual = nvfp4_grouped_mm._to_nvfp4_rht_rs_then_scaled_grouped_mm(
        x,
        weight,
        tuple(1 if i % 2 == 0 else -1 for i in range(16)),
        torch.tensor([1234], dtype=torch.int64, device="cuda"),
        offs=offsets,
        pad_token_groups_for_grouped_mm=True,
        kernel_preference=kernel_preference,
    )
    expected.backward(dy)
    actual.backward(dy)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(weight.grad[:first_active]) == 0
    assert torch.count_nonzero(weight.grad[first_active + 8 :]) == 0
    assert compute_error(expected, actual) >= 15.0
    assert compute_error(x_ref.grad, x.grad) >= 14.0
    active = slice(first_active, first_active + 8)
    assert compute_error(weight_ref.grad[active], weight.grad[active]) >= 12.0
