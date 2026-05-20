import torch


_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None

_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None

_GLOBAL_RANK = None
_WORLD_SIZE = None


def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def model_parallel_is_initialized():
    return (
        _MODEL_PARALLEL_GROUP is not None
        and _TENSOR_MODEL_PARALLEL_GROUP is not None
        and _DATA_PARALLEL_GROUP is not None
    )


def initialize_model_parallel(tensor_model_parallel_size_=1):
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _GLOBAL_RANK
    global _WORLD_SIZE

    assert torch.distributed.is_initialized(), "torch.distributed must be initialized first."
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    ensure_divisibility(world_size, tensor_model_parallel_size)

    data_parallel_size = world_size // tensor_model_parallel_size
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    _GLOBAL_RANK = rank
    _WORLD_SIZE = world_size

    all_data_parallel_group_ranks = []

    for i in range(tensor_model_parallel_size):
        ranks = range(i, world_size, tensor_model_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i] for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group


def destroy_model_parallel():
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    global _GLOBAL_RANK
    global _WORLD_SIZE

    _MODEL_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    _GLOBAL_RANK = None
    _WORLD_SIZE = None


def get_model_parallel_group():
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_model_parallel_world_size():
    return get_tensor_model_parallel_world_size()


def set_tensor_model_parallel_rank(rank):
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_model_parallel_rank():
    return get_tensor_model_parallel_rank()


def get_data_parallel_world_size():
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_tensor_model_parallel_src_rank():
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_global_rank():
    assert _GLOBAL_RANK is not None, "global rank is not initialized"
    return _GLOBAL_RANK


def get_world_size():
    assert _WORLD_SIZE is not None, "world size is not initialized"
    return _WORLD_SIZE


def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


def _reduce(input_):
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_


def _split(input_):
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    last_dim = input_.dim() - 1
    last_dim_size = divide(input_.size(last_dim), world_size)
    input_list = torch.split(input_, last_dim_size, dim=last_dim)
    rank = get_tensor_model_parallel_rank()
    return input_list[rank].contiguous()


def _gather(input_):
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())
    return torch.cat(tensor_list, dim=last_dim).contiguous()


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d].clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        loss = torch.log(sum_exp_logits) - predicted_logits
        softmax = exp_logits / sum_exp_logits.unsqueeze(dim=-1)
        ctx.save_for_backward(softmax, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        grad_input = softmax
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)
