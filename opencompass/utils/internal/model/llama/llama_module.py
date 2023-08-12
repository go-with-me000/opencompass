from torch import nn
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
import torch.nn.functional as F
import torch
try:
    from xformers.ops import swiglu
    from xformers.ops.swiglu_op import SwiGLUFusedOp
except:  # 防止没有安装直接就没办法 import 了
    pass
from flash_attn.ops.fused_dense import all_reduce


class FeedForward(nn.Module):
    def __init__(
        self,
        # dim: int,
        # hidden_dim: int,
        in_features, hidden_features, out_features=None, activation='gelu_approx',
        process_group = None, bias=True, sequence_parallel=False, checkpoint_lvl=0, 
        heuristic='auto', device=None, dtype=None,
        multiple_of: int = 256  # 是多少的倍数
    ):
        super().__init__()
        # hidden_dim = int(2 * in_features / 3)
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            # in_features, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w2 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w3 = RowParallelLinear(
            # hidden_dim, out_features, bias=False, input_is_parallel=True, init_method=lambda x: x
            hidden_features, out_features, process_group, bias = bias, sequence_parallel=False, device=device, dtype=dtype
        )

    def forward(self, x):
        # out = swiglu(x, self.w1.weight, self.w1.bias, self.w2.weight, self.w2.bias, self.w3.weight, self.w3.bias, op=SwiGLUFusedOp)
        # out = all_reduce(out, self.w3.process_group)
        # return out
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # return rms_norm(x, self.weight, self.eps)  # 这里没用上，是由于它只支持特定 size 的 hidden size
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
