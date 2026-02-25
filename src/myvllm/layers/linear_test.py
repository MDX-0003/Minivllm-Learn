import torch
from torch import dist, nn

class Dist:
    def __init__(self,world_size :int):
        self._world_size = world_size
        self._rank_pool  = set(range(world_size))
    def get_world_size(self):
        return self._world_size
    def get_rank(self):
        return self._rank_pool.pop()
dist = Dist(world_size=4)#写在这里，在不改变Linear clas的前提下，劫持其中的dist来源

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim #tp从哪个维度切分，0行并行1列并行
        self.tp_rank = dist.get_rank()#当前设备在tp维度上的rank，即排第几号
        self.tp_size = dist.get_world_size()#tp维度的总大小，即tp并行的设备数量
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.zeros(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        #初始化一个权重加载器，初始为none,需要子类实现具体的加载方法

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")


class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    #切分的核心发生在这里，对于一个batch的权重矩阵，需要将合适的部分切出来放进当前设备的参数中
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0) #由于权重是[out,in]，所以取0维
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size#对out整除设备数
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)#从full_weight中切出长度为share_size行
        param_data.copy_(slided_weight)#拷贝到当前设备的nn.Parameter中

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


def print_weight(layers):
    print(f"Before loading weights:")
    print(layers.weight)
    layers.weight.weight_loader(layers.weight, full_weight)
    print(f"After loading weights:")
    print(layers.weight)
if __name__ == "__main__":
    
    input_size = 16
    output_size = 32
    world_size = dist.get_world_size()

    share_out = output_size // world_size
    shards = []
    for i in range(world_size):
        shard = torch.full((share_out, input_size),
                           fill_value= i + 1 ,
                           dtype=torch.float32)
        shards.append(shard)
    full_weight = torch.cat(shards, dim=0).cuda()
    print("full_weight:", full_weight.shape)


    for _ in range(world_size):
        layer = ColumnParallelLinear(input_size, output_size).cuda()
        print_weight(layer)