import math
import torch
import os


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                self.eta_min
                + (1 - self.eta_min)
                * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                / 2
            )

        else:
            return self.eta_min + self.lr_step * step


class WarmLinear:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup_step = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup_step = warmup_step
            self.warmup_lr_step = (1 - eta_min) / warmup_step
        self.decay_lr_step = (eta_min - 1) / (tmax - self.warmup_step)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup_step:
            return max(self.eta_min, 1 + self.decay_lr_step * (step - self.warmup_step))
        else:
            return max(self.eta_min, self.eta_min + self.warmup_lr_step * step)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {} local rank {}): {}".format(
            args.rank, args.local_rank, "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
