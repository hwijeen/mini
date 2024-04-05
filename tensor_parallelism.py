import torch


# no parallelism
X = torch.tensor(  # 2x4
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
)
A = torch.tensor(  # 4x2
    [
        [10, 14],
        [11, 15],
        [12, 16],
        [13, 17],
    ]
)
Y = X @ A

# split A column-wise
A1, A2 = torch.tensor_split(A, 2, dim=1)
Y1 = X @ A1  # X is colpied to each device
Y2 = X @ A2
Y_col = torch.cat([Y1, Y2], dim=1)
assert torch.allclose(Y, Y_col)

# split A row-wise
X_1, X_2 = torch.tensor_split(X, 2, dim=1)  # column-wise split
A_1, A_2 = torch.tensor_split(A, 2, dim=0)  # row-wise split
Y_row = X_1 @ A_1 + X_2 @ A_2
assert torch.allclose(Y, Y_row)


# Now actually run it on multiple devices
import os
import torch.distributed as dist
import torch.multiprocessing as mp  # we don't rely on torchrun


def run_mm_parallel(rank, fn, X, backend="nccl"):
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend, rank=rank, world_size=2)

    fn(rank, X)


def column_parallel(rank, X):
    print(f"Ruuning in rank {rank}")

    X = X.to(f"cuda:{rank}")
    print(f"Rank {rank} has X: {X}")

    if rank == 0:
        W = torch.tensor([[10], [11], [12], [13]], device=f"cuda:{rank}").float()
    elif rank == 1:
        W = torch.tensor([[14], [15], [16], [17]], device=f"cuda:{rank}").float()

    XW = X @ W
    print(f"Rank {rank} has XW: {XW}")

    group = dist.new_group([0, 1])
    result = [torch.zeros_like(XW) for _ in range(2)]
    dist.all_gather(result, XW, group=group)
    print(f"Rank {rank} has gathered result: {result}")
    return torch.cat(result, dim=1)


if __name__ == "__main__":
    processes = []
    X = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]).float()

    mp.spawn(run_mm_parallel, args=(column_parallel, X), nprocs=2)

    # mp.set_start_method("spawn")
    # for rank in range(2):
    #     p = mp.Process(target=run_mm_parallel, args=(rank, column_parallel, X))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()  # blocking

#####
# Ruuning in rank 1
# Ruuning in rank 0
# Rank 1 has X: tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]], device='cuda:1')
# Rank 0 has X: tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]], device='cuda:0')
# Rank 1 has XW: tensor([[ 98.],
#         [346.]], device='cuda:1')
# Rank 0 has XW: tensor([[ 74.],
#         [258.]], device='cuda:0')
# Rank 1 has gathered result: [tensor([[ 74.],
#         [258.]], device='cuda:1'), tensor([[ 98.],
#         [346.]], device='cuda:1')]
# Rank 0 has gathered result: [tensor([[ 74.],
#         [258.]], device='cuda:0'), tensor([[ 98.],
#         [346.]], device='cuda:0')]
#####
