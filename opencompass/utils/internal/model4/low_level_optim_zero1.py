import copy
import os
import sys
from functools import partial

import torch
import torch.distributed as dist
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.constants import IS_TENSOR_PARALLEL
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.tensor import ColoTensor
from colossalai.tensor.distspec import DistPlacementPattern
from colossalai.utils import is_using_pp
from colossalai.utils.common import (
    _calc_l2_norm,
    _calc_lp,
    _get_tensor_norm,
    _move_norm_to_cuda,
)
from colossalai.utils.cuda import get_current_device
from colossalai.utils.megatron_timers import timer
from colossalai.zero.sharded_optim._utils import (
    flatten,
    get_grad_accumulate_object,
    has_inf_or_nan,
    release_param_grad,
    sync_param,
)
from colossalai.zero.sharded_optim.bookkeeping import (
    BucketStore,
    GradientStore,
    ParameterStore,
)
from colossalai.zero.sharded_optim.low_level_optim import LowLevelZeroOptimizer
from torch._six import inf
from torch.optim import Optimizer

from model.optim_stuff.initializer_node import init_intra_node_process_group
from utils.checkpoint_utils import try_get_device
from utils.storage_manager import get_storage_manager
from utils.utils import (
    get_dp_rank,
    get_pp_rank,
    get_pp_size,
    get_process_rank_utils,
    get_tp_rank,
    get_tp_size,
)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class SplitPlan:
    greedy = "greedy"
    local = "local"


def get_fns(folder):
    return get_storage_manager().get_fns(folder)


def load_states(fp, *args, stype="torch", **kwargs):
    return get_storage_manager().load(fp, *args, stype=stype, **kwargs)


INTRA_GPU_NUMS = 8


class NoPPModifiedLowLevelZeroOptimizer(LowLevelZeroOptimizer):
    """
    Modified Low Level Zero Optimizer without pipeline parallel, and support overlap_communication.
    """

    def __init__(  # pylint: disable=W0231
        self,
        optimizer: Optimizer,
        # grad scaler config
        initial_scale=2**16,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        hysteresis=2,
        max_scale: int = 2**24,
        # grad clipping
        clip_grad_norm=0.0,
        verbose=False,
        # communication
        reduce_bucket_size=1024 * 1024,
        communication_dtype=None,
        overlap_communication=False,
        # stage 2
        partition_grad=False,
        dp_parallel_mode=ParallelMode.DATA,
        mp_parallel_mode=ParallelMode.MODEL,
        # cpu offload
        cpu_offload=False,
        # forced dtype
        forced_dtype=None,
        split_type=SplitPlan.greedy,
        overlap_broadcast=False,
    ):
        assert partition_grad is False, "unsupport zero2 or zero3"
        # assert (
        #     gpc.get_world_size(ParallelMode.MODEL) == 1
        # ), f"only support DDP by now! but got ParallelMode.MODEL size ==={gpc.get_world_size(ParallelMode.MODEL)}"
        ColossalaiOptimizer.__init__(self, optim=optimizer)  # pylint: disable=W0233
        if is_using_pp() and overlap_communication:
            raise RuntimeError(
                "The pipeline parallelism is not compatible with overlap_communication, "
                "please set overlap_communication=False if you want to use the pipeline parallelism."
            )
        if is_using_pp() and partition_grad:
            raise RuntimeError(
                "The pipeline parallelism is not compatible with Zero2, "
                "please set partition_grad=False if you want to use the pipeline parallelism."
            )

        assert partition_grad is False, "NoPPModifiedLowLevelZeroOptimizer not support partition_grad by now."

        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose
        self.using_pp = is_using_pp()

        # stage 2
        self._partition_grads = partition_grad

        # cpu_offload
        self._cpu_offload = cpu_offload

        # get process groups
        self.split_type = split_type
        self._mp_parallel_mode = mp_parallel_mode
        self._dp_group = gpc.get_group(dp_parallel_mode)
        self._dp_parallel_mode = dp_parallel_mode

        if self.split_type == SplitPlan.local:
            # self._dp_parallel_mode = ParallelMode.INTRA_NODE
            init_intra_node_process_group()
            self._world_size = INTRA_GPU_NUMS // gpc.get_world_size(ParallelMode.MODEL)
            self._local_rank = gpc.get_local_rank(ParallelMode.INTRA_NODE)
            assert self._local_rank < self._world_size
            self._broadcast_group = gpc.get_group(ParallelMode.INTRA_NODE)
            self._broadcast_parallel_mode = ParallelMode.INTRA_NODE
            self._zero_parallel_size = INTRA_GPU_NUMS
            assert self._world_size <= 8, f"{self._world_size} must <= 8"
        else:
            self._local_rank = gpc.get_local_rank(dp_parallel_mode)
            self._world_size = gpc.get_world_size(dp_parallel_mode)
            self._broadcast_group = gpc.get_group(dp_parallel_mode)
            self._broadcast_parallel_mode = self._dp_parallel_mode
            self._zero_parallel_size = gpc.data_parallel_size

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(self._broadcast_parallel_mode)
        self._grad_store = GradientStore(self._dp_parallel_mode)
        self._bucket_store = BucketStore(self._dp_parallel_mode)

        if gpc.is_initialized(mp_parallel_mode) and gpc.get_world_size(mp_parallel_mode) > 1:
            self._mp_group = gpc.get_group(mp_parallel_mode)
        else:
            self._mp_group = None

        # fp16 and fp32 params for mixed precision training
        self._fp16_param_groups = dict()
        self._fp32_flat_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            verbose=verbose,
        )
        self._found_overflow = torch.FloatTensor([0]).to(get_current_device())

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group["params"]
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # (WGT): We need to record the rank in which parameter groups are not assigned parameters.
        self.param_group_has_params = []
        self.param_group_no_params_ranks = []
        self.padding_grad = torch.zeros([32], dtype=self._dtype, device=get_current_device())
        self.padding_tensor = torch.zeros([32], dtype=self._dtype, device=get_current_device())

        self.pj_step_count = 0
        self.sync_count = 0
        self.reduce_count = 0
        self.last_reduce_step = 0
        self.rank_unique_id = f"gpus-{gpc.get_world_size(ParallelMode.GLOBAL)}\
_pp-{get_pp_rank()}_tp-{get_tp_rank()}_dp-{self.get_local_rank()}.pickle"
        # self.params_per_rank_id_dict = [[] for _ in range(self._world_size)]
        # self.split_type = SplitPlan.local
        self.overlap_broadcast = overlap_broadcast

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups[group_id] = group_params

            # assign parameters to ranks
            # the params in the list are sorted
            params_per_rank, no_params_ranks = self._partition_param_list(group_params)
            self.param_group_no_params_ranks.append(no_params_ranks)
            self.param_group_has_params.append(self._local_rank not in no_params_ranks)

            # store the mapping between param to rank
            # each param should belong to only one rank
            for rank, params in enumerate(params_per_rank):
                # (WGT): Check whether any rank is not assigned params.
                if len(params) != 0:
                    self._param_store.add_fp16_param_list_by_rank_group(rank, group_id, params)
                    for param in params:
                        self._param_store.set_param_to_rank(param, rank)

            # move to cpu to make room to create the flat tensor
            # move_tensor(params, device='cpu')
            for param in group_params:
                param.data = param.data.cpu()

            # flatten the reordered tensors
            for rank in range(self._world_size):
                # (WGT): No flat fp16 buffer is allocated if the process has no parameters.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                    with torch.no_grad():
                        flat_tensor = flatten(tensor_list)
                    flat_tensor = flat_tensor.data.cuda()
                    self._param_store.add_flat_fp16_param_by_rank_group(rank, group_id, flat_tensor)
                    sync_param(flat_tensor=flat_tensor, tensor_list=tensor_list)

            # create a copy of fp32 weights of the parameters for which this rank is responsible
            # (WGT): No flat fp32 buffer is allocated if the process has no parameters.
            if self.param_group_has_params[group_id]:
                fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(self._local_rank, group_id)
                fp32_flat_current_rank = fp16_flat_current_rank.float()
                device = "cpu" if self._cpu_offload else get_current_device()
                fp32_flat_current_rank = fp32_flat_current_rank.to(device)
                fp32_flat_current_rank.requires_grad = True
                self._fp32_flat_param_groups_of_current_rank[group_id] = fp32_flat_current_rank

                # need to replace the params in the `params` field in the optimizer
                # so that when the optimizer calls step(), it only updates the tensors
                # managed by this data parallel rank
                param_group["params"] = [fp32_flat_current_rank]

            # set reduction state
            for param in self._fp16_param_groups[group_id]:
                self._param_store.set_param_reduction_state(param, False)
        assert len(self._fp16_param_groups) != 0

        # (WGT): If a rank is not assigned any arguments, 'has_params' is False.
        self.has_params = sum(self.param_group_has_params) != 0

        # intialize communication stream for
        # communication-compuation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_hook()

    def get_broadcast_size(self):
        """
        Get the size of data parallel, if not, it will be 1.
        """
        return gpc.get_world_size(self._broadcast_parallel_mode)

    def get_local_rank(self):
        """
        Get the size of data parallel, if not, it will be 1.
        """
        return self._local_rank

    def _partition_param_list(self, param_list):
        no_params_ranks = []
        params_per_rank = [[] for _ in range(self._world_size)]
        numel_per_rank = [0 for _ in range(self._world_size)]
        self.params_per_rank_id_dict = [[] for _ in range(self._world_size)]

        sorted_params = sorted(param_list, key=lambda x: x.numel(), reverse=True)
        for i, param in enumerate(sorted_params):
            global_id = str(i)
            for j in range(len(param.size())):
                global_id = "_".join([global_id, str(param.size()[j])])

            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].append(param)
            self.params_per_rank_id_dict[rank_to_go].append(global_id)
            numel_per_rank[rank_to_go] += param.numel()

        # (WGT): Check whether any rank is not assigned to parameters.
        for rank, params in enumerate(params_per_rank):
            if len(params) == 0:
                no_params_ranks.append(rank)

        if self._verbose:
            if self.split_type == SplitPlan.local:
                self._logger.info(
                    f"Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}",
                    parallel_mode=self._broadcast_parallel_mode,
                )
            else:
                self._logger.info(
                    f"Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}",
                    ranks=[0],
                    parallel_mode=self._dp_parallel_mode,
                )

        return params_per_rank, set(no_params_ranks)

    def backward(self, loss, retain_graph=False):
        if self.using_pp:
            return super().backward(loss, retain_graph)
        else:
            return self.nopp_backward(loss, retain_graph)

    def nopp_backward(self, loss, retain_graph=False):
        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

    def _has_inf_or_nan(self, tensor):
        try:
            tensor_mean = float(tensor.mean())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if tensor_mean == float("inf") or tensor_mean == -float("inf"):
                return True
            return False

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            # (WGT): The following operations are performed only on the rank to which parameters are assigned.
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                    if avg_grad is not None and has_inf_or_nan(avg_grad):
                        self._found_overflow.fill_(1.0)
                        break

        # all-reduce over MODEL ranks
        # dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MODEL))

        # all-reduce over all ranks
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.GLOBAL))

        return self._found_overflow.item() > 0

    def _step(self, closure=None):
        # nopp 和 pp 的step操作应该没啥区别？
        return self.nopp_step(closure)

    def nopp_step(self, closure=None):
        assert closure is None, "closure is not supported by step()"
        self.pj_step_count += 1

        # check for overflow
        found_inf = self._check_overflow()
        # Because you may encounter inf when computing norm
        timer("cal_norm").start()
        norm_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
                parameters = self._param_store.get_fp16_params_by_rank_group(group_id=group_id, rank=self._local_rank)
            else:
                # (WGT): In order to prevent collection communication from hanging,
                # we need to involve rank that are not assigned parameters in compute_norm(),
                # so we give them a fp16 vector of 0 values.
                gradients = [self.padding_grad]
                parameters = [self.padding_tensor]

            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = no_pp_compute_norm(
                    gradients=gradients,
                    parameters=parameters,
                    reduce_group=self._broadcast_group,  # 由于已经在节点之间同步了，因此在节点内就行了
                )
                if norm_group == -1:
                    timer("cal_norm").stop()
                    found_inf = True
                    break
                norm_groups.append(norm_group)

        loss_scale = float(self.loss_scale.item())  # backup
        self.grad_scaler.update(found_inf)
        # update loss scale if overflow occurs
        if found_inf:
            from utils.monitor_and_alert import get_process_rank, send_alert_message

            if get_process_rank() == 0:
                send_alert_message(message="Overflow occurs, please check it.")
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, None

        # copy the grad of fp16 param to fp32 param
        single_grad_partition_groups = []
        global_norm = 0
        for group_id in range(self.num_param_groups):
            # compute norm
            # (WGT): The following operations are performed only on the rank to which parameters are assigned.
            if not self.param_group_has_params[group_id]:
                continue
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)

            # create flat gradient for the flat fp32 params
            fp16_avg_grads = gradients
            flat_fp16_avg_grads = flatten(fp16_avg_grads)

            dtype = self._fp32_flat_param_groups_of_current_rank[group_id].dtype
            flat_fp32_avg_grads = flat_fp16_avg_grads.to(dtype)

            param_shape = self._fp32_flat_param_groups_of_current_rank[group_id].shape
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm > 0:
            global_norm = sum(norm_groups) ** 0.5

        # (WGT): The following operations are performed only on the rank to which parameters are assigned.
        if len(single_grad_partition_groups) != 0:
            self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)

        timer("cal_norm").stop()
        # update the parameters
        timer("step").start()

        # to avoid modify engine.update(), we use envvar to pass arguments
        enable_skip = os.environ.get("ENABLE_SKIP_PARAM_UPDT", "False")
        if enable_skip == "True":
            grad_norm_baseline = float(os.environ.get("GRAD_NORM_BASE"))
            grad_norm_max = float(os.environ.get("GRAD_NORM_MAX"))
            grad_norm_ref = max(grad_norm_baseline, grad_norm_max)
            if (global_norm / loss_scale) > grad_norm_ref:
                # skip weight update if normalized gradient increased steeply
                timer("step").stop()
                from utils.logger import LLM_LOGGER as logger

                logger.warning(
                    f"skip weight update because normalized "
                    f"gradient({global_norm/loss_scale}) > reference ({grad_norm_ref}).",
                    ranks=[0],
                )
                # encode grad_norm as -99.0 to indicate this case
                return False, -99.0

        # (WGT): For those ranks that are not assigned parameters, we just wait for other ranks
        # to send them updated their own parameters.
        if self.has_params:
            self.optim.step()
            # release the fp32 grad
            release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())
            # update fp16 partition updated by the current rank
            for group_id in range(len(self._fp16_param_groups)):
                if self.param_group_has_params[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                        rank=self._local_rank, group_id=group_id
                    )
                    fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                    fp16_param.data.copy_(fp32_param)

        # broadcast the updated model weights

        # def print_GPU_memory(index=0):
        #     import pynvml
        #     handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        #     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #     print(f'GPU {index}: Tota GPU ME:', meminfo.total/1024**3, 'GB') #总的显存大小
        #     print(f'GPU {index}: Used GPU ME:', meminfo.used/1024**3, 'GB')  #已用显存大小
        #     print(f'GPU {index}: Free GPU ME:', meminfo.free/1024**3, 'GB')  #剩余显存大小

        self.broadcast_params(overlap=False)

        timer("step").stop()
        # update gradients may not be needed here, because the sync_params function is used in initialization,
        # so synchronization is maintained
        return True, global_norm / loss_scale

    @staticmethod
    def do_reduce_and_sync_grad():
        # step_count = int(os.environ["STEP_COUNT"])
        # accumulate_grad = int(os.environ["GRAD_ACC_NUM"])

        # if accumulate_grad == 1:
        #     # When args.accumulate_grads == 1, we always sync grad.
        #     return True
        # else:
        #     return (step_count + 1) % accumulate_grad == 0
        return os.environ["last_acc_step"] == "1"

    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._fp16_param_groups[group_id]
            for param in param_group:
                if param.requires_grad:
                    # determines the reduction destionation rank
                    # this is only valid for stage 2
                    # dst_rank = None means using all-reduce
                    # else using reduce
                    if self._partition_grads:
                        reduce_rank = self._param_store.get_param_rank(param)
                    else:
                        reduce_rank = None

                    def _define_and_attach(param, reduce_rank):
                        # get the AccumulateGrad object of the param itself
                        accum_grad_obj = get_grad_accumulate_object(param)
                        self._grad_store.add_accumulate_grad_object(accum_grad_obj)

                        reduction_func = partial(
                            self._reduce_and_remove_grads_by_bucket, param=param, reduce_rank=reduce_rank
                        )

                        # define hook
                        # NOT IMPORTANT BUT GOOD TO KNOW:
                        # args here is not grad, but allow_unreacable and accumulate_grad
                        def reduce_grad_hook(*args):  # pylint: disable=W0613
                            # (WGT): Skip reduce hook when accumulate_grad is triggered.
                            # accumulate_grad is the command line parameter in utils.py,
                            # not to be confused with accumulate_grad in low_level_optimzer.
                            if NoPPModifiedLowLevelZeroOptimizer.do_reduce_and_sync_grad():
                                now_step = int(os.environ["setp_count"])
                                if now_step != self.last_reduce_step:
                                    self.reduce_count += 1
                                    self.last_reduce_step = now_step

                                reduction_func()

                        accum_grad_obj.register_hook(reduce_grad_hook)

                    _define_and_attach(param, reduce_rank)

    def broadcast_params(self, overlap=False):
        handles = []

        for group_id in range(self.num_param_groups):
            for rank in range(self._world_size):
                # (WGT): The following operations are performed only on the rank to which parameters are assigned.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=rank, group_id=group_id)
                    # grank = gpc.get_ranks_in_group(group_type)[rank]  # need to convert to the global rank
                    # assert grank == rank, f"{grank} == {rank}"
                    g_rank = gpc.get_ranks_in_group(self._broadcast_parallel_mode)[rank]
                    handle = dist.broadcast(fp16_param, src=g_rank, group=self._broadcast_group, async_op=True)
                    handles.append(handle)

        if not overlap:
            for handle in handles:
                handle.wait()
        else:
            return handles

    def sync_grad(self):
        self.sync_count += 1
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, _ in reduction_states.items():
            reduction_states[tensor] = False

        # accumulate gradient
        avg_gradients = self._grad_store._averaged_gradients
        for group_id in range(self.num_param_groups):
            # (WGT): The following operations are performed only on the rank to which parameters are assigned.
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                param_group = self._param_store.get_fp16_params_by_rank_group(self._local_rank, group_id)

                if group_id not in avg_gradients:
                    avg_gradients[group_id] = []

                param_idx = 0
                for param in param_group:
                    if param.grad is not None:
                        if len(avg_gradients[group_id]) == param_idx:
                            avg_gradients[group_id].append(param.grad)
                        else:
                            avg_gradients[group_id][param_idx].add_(param.grad)
                        param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    # TODO 需要加入state_dict方法
    def state_dict(self):
        states = {}
        grad_scaler = self.grad_scaler.state_dict()
        # TODO 需要考虑没有 grad_scaler 的情况
        states["grad_scaler"] = grad_scaler

        # 传入的 optimizer 的 state , todo 如果这个optimizer还没跑过就state dict的话，可能会导致其中的一些东西不存在，可能会在之后报错？
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        # 自身管理的 fp32 的权重部分
        flat_fp32_weights = {}
        for group_id, param in self._fp32_flat_param_groups_of_current_rank.items():
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                assert param.grad is None
                flat_fp32_weights[group_id] = param
        states["flat_fp32_weights"] = flat_fp32_weights

        # TODO 应该还需要有一些sanity check的内容

        # 存下 optimizer 的保存信息
        states["zero_devide_optim_plan"] = self.params_per_rank_id_dict

        # 如果是

        # TODO 需要考虑出现 dp 数量变化的情况

        return states

    def load_state_dict(self, states):
        # TODO 需要考虑出现 dp 数量变化的情况

        # TODO 需要考虑没有 loss_scaler 的情况
        grad_scaler = states["grad_scaler"]
        self.grad_scaler.load_state_dict(grad_scaler)

        # load optimizer
        optim_states = states["base_optim_states"]
        self.optim.load_state_dict(optim_states)

        # fp32 权重
        flat_fp32_weights = states["flat_fp32_weights"]
        assert set(flat_fp32_weights.keys()) == set(self._fp32_flat_param_groups_of_current_rank)
        for group_id, param in flat_fp32_weights.items():
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                _param = self._fp32_flat_param_groups_of_current_rank[group_id]
                assert _param.shape == param.shape
                _param.data.copy_(param.data)

        # 需要对model的进行赋值
        for group_id in range(len(self._fp16_param_groups)):
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                    rank=self._local_rank, group_id=group_id
                )
                fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                fp16_param.data.copy_(fp32_param)  # 自动也就改变了 model 那边的值了

        # 读下optimizer 的保存信息
        # TODO: check 下两次 optimzer 参数分配的策略是否是一样的
        if "zero_devide_optim_plan" in states:
            self.params_per_rank_id_dict = states["zero_devide_optim_plan"]

    def load_optimizer_checkpoint(self, folder, states=None):
        if states is None:
            fns = get_fns(folder)
            max_tp, max_pp, max_dp = 0, 0, 0
            for fn in fns:
                if fn.startswith("optimizer_") and not fn.endswith(".md5"):
                    _, tp, pp, dp = os.path.splitext(fn)[0].split("_")
                    max_dp = max(max_dp, int(dp[2:]))
                    max_tp = max(max_tp, int(tp[2:]))
                    max_pp = max(max_pp, int(pp[2:]))

            assert (
                self.get_broadcast_size() == max_dp + 1
            ), f"The weights are save for {max_dp+1} data parallel, while current has\
{self.get_broadcast_size()} zero broadcast range (model/low_level_optim_zero1.py)."
            assert (
                get_pp_size() == max_pp + 1
            ), f"The weights are save for {max_pp+1} pipelines, while current has {get_pp_size()} pipelines"
            assert (
                get_tp_size() == max_tp + 1
            ), f"The weights are save for {max_tp+1} parallelism, while current has {get_tp_size()} tensor parallelism"

            fp = f"optimizer_tp{get_tp_rank()}_pp{get_pp_rank()}_dp{self.get_local_rank()}.pt"
            should_load_fp = os.path.join(folder, fp)
            states = load_states(should_load_fp, map_location=try_get_device())

            fp_meta = os.path.join(folder, self.rank_unique_id)
            try:
                zero_devide_optim_plan = load_states(fp_meta, stype="pickle")
                states.update({"zero_devide_optim_plan": zero_devide_optim_plan})

                # TODO: santiy check
            except Exception as e:
                from utils.logger import LLM_LOGGER

                LLM_LOGGER.warning(
                    f"Read zero optimzer split file '{fp_meta}', for '{e}', \
Please check whether ckpts are saved with the model/low_level_optim_zero1.py interface."
                )
                # raise e

        self.load_state_dict(states)

    def save_optimizer_checkpoint(self, folder, handler):
        """
        Each shard saves its own
        - folder
            - optimizer_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.pt
        """

        # TODO sanity check for optimizer type
        states = self.state_dict()
        # 只选择广播范围内的rank进行保存ckpt
        save_range = self._zero_parallel_size
        # zero 最大的切分范围就是 DP ，
        if folder is not None and gpc.get_local_rank(ParallelMode.DATA) < save_range:
            fp = f"optimizer_tp{get_tp_rank()}_pp{get_pp_rank()}_dp{self.get_local_rank()}.pt"
            fp = os.path.join(folder, fp)
            if "zero_devide_optim_plan" in states:
                # 如果存在zero切分的元数据信息，我们也保存下来
                params_per_rank_id_dict = states.pop("zero_devide_optim_plan")
                fp_meta = os.path.join(folder, self.rank_unique_id)
                handler.save(fp_meta, params_per_rank_id_dict, stype="pickle")
            handler.save(fp, states)
        else:
            states = copy.deepcopy(states)
            return states


def is_model_parallel_parameter(p):
    # colossalai.utils.common.is_model_parallel_parameter cannot process the ColoParam
    return (hasattr(p, IS_TENSOR_PARALLEL) and getattr(p, IS_TENSOR_PARALLEL)) or (
        isinstance(p, ColoTensor) and p.dist_spec.placement == DistPlacementPattern.SHARD
    )


def no_pp_compute_norm(gradients, parameters, reduce_group, norm_type=2):
    """Get the norm
    Arguments:
        gradients (Iterable[Tensor]): The gradient value
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters, need total_norm**(1/norm) before using.
    """

    enable_cuda_kernels = gradients[0].device.type == "cuda"
    # Norm parameters.
    norm_type = float(norm_type)

    has_zero_shared_param: bool = False
    for param in parameters:
        if hasattr(param, "colo_attr") and param.colo_attr.sharded_data_tensor.is_sharded:
            raise RuntimeError("Currently not support Zero3")

    # Parameters can be on CPU or CUDA
    # If parameters are on CPU, disable CUDA kernerls

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.FloatTensor([float(total_norm)]).to(gradients[0].device)
        # Take max across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL) and gpc.get_world_size(ParallelMode.MODEL) > 1:
            dist.all_reduce(
                total_norm_cuda, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MODEL), async_op=False
            )
        if has_zero_shared_param:
            dist.all_reduce(
                total_norm_cuda, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.DATA), async_op=False
            )
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = []
        for g, p in zip(gradients, parameters):
            # TODO consider the pipeline shared parameter

            if (
                gpc.is_initialized(ParallelMode.PIPELINE)
                and hasattr(p, "pipeline_shared_module_pg")
                and dist.get_rank(p.pipeline_shared_module_pg) == 0
            ):  # if shared between different pipe, only count o
                tensor_parallel_grads.append(g.data.float())
                # if is_model_parallel_parameter(p):
                #     tensor_parallel_grads.append(g.data.float())
                # else:
                #     no_tensor_parallel_grads.append(g.data.float())
            elif (
                gpc.is_initialized(ParallelMode.PIPELINE)
                and hasattr(p, "pipeline_shared_module_pg")
                and dist.get_rank(p.pipeline_shared_module_pg) != 0
            ):
                continue
            elif (
                gpc.is_initialized(ParallelMode.TENSOR)
                and not is_model_parallel_parameter(p)
                and gpc.get_local_rank(ParallelMode.TENSOR) == 0
            ):  # if not used in each chunk, such as layernorm
                # no_tensor_parallel_grads.append(g.data.float())
                tensor_parallel_grads.append(g.data.float())
            elif is_model_parallel_parameter(p):
                # reductor = (gpc.get_world_size(ParallelMode.TENSOR) / getattr(p, NUM_PARTITIONS))**(1 / norm_type)
                tensor_parallel_grads.append(g.data.float())
            elif gpc.get_local_rank(ParallelMode.TENSOR) != 0:
                continue
            else:
                raise RuntimeError("Should not arrive here")

        if norm_type == 2.0 and enable_cuda_kernels:
            tensor_parallel_norm = _calc_l2_norm(tensor_parallel_grads) ** norm_type
            # no_tensor_parallel_norm = _calc_l2_norm(no_tensor_parallel_grads)**norm_type
        else:
            tensor_parallel_norm = _calc_lp(tensor_parallel_grads, norm_type)
            # no_tensor_parallel_norm = _calc_lp(no_tensor_parallel_grads, norm_type)

        # If norm is type of float, then we convert them into torch.Tensor.
        tensor_parallel_norm = _get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
        # no_tensor_parallel_norm = _get_tensor_norm(no_tensor_parallel_norm, enable_cuda_kernels)
        # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
        if not enable_cuda_kernels:
            tensor_parallel_norm = _move_norm_to_cuda(tensor_parallel_norm)
            # no_tensor_parallel_norm = _move_norm_to_cuda(no_tensor_parallel_norm)

        # total_norm = tensor_parallel_norm + no_tensor_parallel_norm
        total_norm = tensor_parallel_norm

        # Sum across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL):
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.MODEL))

        # This is because we use zero1, so we need to use this reduction.
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=reduce_group)

        if torch.is_tensor(total_norm):
            total_norm = total_norm.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    return total_norm
