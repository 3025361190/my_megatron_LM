import contextlib
import random

import numpy as np
import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from distributed import get_tensor_model_parallel_rank


_TENSOR_MODEL_PARALLEL_RNG_TRACKER_NAME = "tensor-model-parallel-rng"


def _set_cuda_rng_state(new_state, device=-1):
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):

        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStatesTracker:
    def __init__(self):
        self.states_ = {}
        self.seeds_ = set()

    def reset(self):
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        return {name: state for name, state in self.states_.items()}

    def set_states(self, states):
        self.states_ = states

    def add(self, name, seed):
        if seed in self.seeds_:
            raise RuntimeError(f"seed {seed} already exists")
        if name in self.states_:
            raise RuntimeError(f"cuda rng state {name} already exists")

        self.seeds_.add(seed)
        orig_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name):
        if name not in self.states_:
            raise RuntimeError(f"cuda rng state {name} is not added")

        orig_cuda_rng_state = torch.cuda.get_rng_state()
        _set_cuda_rng_state(self.states_[name])
        try:
            yield
        finally:
            self.states_[name] = torch.cuda.get_rng_state()
            _set_cuda_rng_state(orig_cuda_rng_state)


_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    return _CUDA_RNG_STATE_TRACKER


def initialize_random_seed(seed):
    if seed is None or seed <= 0:
        raise ValueError(f"Seed ({seed}) should be a positive integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.device_count() == 0:
        print(f"[rank 0] random seeds initialized on cpu only with seed={seed}", flush=True)
        return

    tensor_model_parallel_seed = seed + 2718 + get_tensor_model_parallel_rank()
    _CUDA_RNG_STATE_TRACKER.reset()
    torch.cuda.manual_seed(seed)
    _CUDA_RNG_STATE_TRACKER.add(_TENSOR_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)
    print(
        f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] "
        f"random seeds ready: default_cuda_seed={seed}, tp_seed={tensor_model_parallel_seed}",
        flush=True,
    )


@contextlib.contextmanager
def fork_tensor_model_parallel_rng():
    with _CUDA_RNG_STATE_TRACKER.fork(_TENSOR_MODEL_PARALLEL_RNG_TRACKER_NAME):
        yield
