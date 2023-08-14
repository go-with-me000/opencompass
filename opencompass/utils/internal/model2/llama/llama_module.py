# type-lint: skip-file
"""
try:
    from xformers.ops import swiglu
    from xformers.ops.swiglu_op import SwiGLUFusedOp
except ImportError:  # avoid ImportError
    pass
from flash_attn.ops.fused_dense import all_reduce
"""
from typing import Optional

import torch
import torch.nn.functional as F
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from torch import nn


class FeedForward(nn.Module):
    """
    FeedForward.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        self.w2 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w3 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        # out = swiglu(
        #     x,
        #     self.w1.weight,
        #     self.w1.bias,
        #     self.w2.weight,
        #     self.w2.bias,
        #     self.w3.weight,
        #     self.w3.bias,
        #     op=SwiGLUFusedOp,
        # )
        # out = all_reduce(out, self.w3.process_group)
        out = self.w3(F.silu(self.w1(x)) * self.w2(x))
        return out


class RMSNorm(nn.Module):
    """
    RMS Normarlization.

    Args:
        dim (int): the dimention of model.
        eps (float): bias term. 1e-6 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
    """

    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
