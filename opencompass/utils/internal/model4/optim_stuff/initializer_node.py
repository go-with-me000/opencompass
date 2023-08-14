#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os

import torch
import torch.distributed as dist
from colossalai.context.parallel_context import ParallelMode, global_context
from colossalai.core import global_context as gpc

from colossalai.context.process_group_initializer import ProcessGroupInitializer
from colossalai.registry import DIST_GROUP_INITIALIZER

INTRA_GPU_NUMS = 8


def try_get_model_world_size():
    try:
        msize = gpc.get_world_size(ParallelMode.MODEL)
        return msize
    except Exception as e:
        raise e


@DIST_GROUP_INITIALIZER.register_module
class Initializer_NodeIntra(ProcessGroupInitializer):
    """A ProcessGroupInitializer for Node Intra comm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert try_get_model_world_size() < INTRA_GPU_NUMS
        assert self.world_size % INTRA_GPU_NUMS == 0, f"{self.world_size} % {INTRA_GPU_NUMS} != 0"
        self.mode_paralledl_size = try_get_model_world_size()
        self.boardcast_intra_worlsize = INTRA_GPU_NUMS // self.mode_paralledl_size
        self.boardcast_intra_stage = INTRA_GPU_NUMS // self.boardcast_intra_worlsize
        self.num_intra_group = self.world_size // self.boardcast_intra_worlsize

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.INTRA_NODE

        for i in range(0, self.num_intra_group, self.boardcast_intra_stage):
            for j in range(self.boardcast_intra_stage):
                start_rank = i * self.boardcast_intra_worlsize + j
                ranks = list(range(start_rank, start_rank + INTRA_GPU_NUMS, self.boardcast_intra_stage))
                # ranks = [i * self.intra_worlsize + j for j in range(self.intra_worlsize)]
                group = dist.new_group(ranks)

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = None
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_NodeInter(ProcessGroupInitializer):
    """A ProcessGroupInitializer for Node Inter comm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.world_size % INTRA_GPU_NUMS == 0, f"{self.world_size} % {INTRA_GPU_NUMS} != 0"
        self.num_intra_group = self.world_size // INTRA_GPU_NUMS

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.INTER_NODE

        ranks = [INTRA_GPU_NUMS * j for j in range(self.num_intra_group)]
        group = dist.new_group(ranks)

        if self.rank in ranks:
            local_rank = ranks.index(self.rank)
            group_world_size = len(ranks)
            process_group = group
            cpu_group = None
            ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


def init_intra_node_process_group():
    world_size = gpc.get_world_size(ParallelMode.GLOBAL)
    g_rank = gpc.get_global_rank()
    initializer = DIST_GROUP_INITIALIZER.get_module("Initializer_NodeIntra")(g_rank, world_size, None, None, None, None)
    parallel_setting = initializer.init_dist_group()
    global_context._register_dist(*parallel_setting)


def init_inter_node_process_group():
    world_size = gpc.get_world_size(ParallelMode.GLOBAL)
    g_rank = gpc.get_global_rank()
    initializer = DIST_GROUP_INITIALIZER.get_module("Initializer_NodeInter")(g_rank, world_size, None, None, None, None)
    parallel_setting = initializer.init_dist_group()
    global_context._register_dist(*parallel_setting)


def test_intra_intre_process_group():
    def get_master_node():
        import subprocess

        if os.getenv("SLURM_JOB_ID") is None:
            raise RuntimeError("get_master_node can only used in Slurm launch!")
        result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
        result = result.decode("utf8").strip()
        return result

    rank = int(os.environ["SLURM_PROCID"])
    locak_rank = rank % 8
    world_size = int(os.environ["SLURM_NPROCS"])
    gpc.init_global_dist(rank, world_size, "nccl", get_master_node(), port=9988)

    # test intra node process group
    init_intra_node_process_group()

    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        gpc.set_device(None)

    gpc.detect_num_processes_on_current_node()
    total_param = torch.tensor([rank]).cuda()
    dist.all_reduce(total_param, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.INTRA_NODE))
    print(f"{total_param}", flush=True)

    # test intre node process group
    init_inter_node_process_group()

    if locak_rank == 0:
        dist.all_reduce(total_param, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.INTER_NODE))
        # print(f"{total_param}", flush=True)


if __name__ == "__main__":
    test_intra_intre_process_group()
