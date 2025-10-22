import torch
import torch.nn as nn
from .utils import InitWeights_He


class BlockRep(nn.Module):
    def __init__(self, ic, oc, k):
        super(BlockRep, self).__init__()
        self.list_module = nn.ModuleList()
        self.cnt = 0
        for i in range(1, k+1, 2):
            self.cnt += 1
            self.list_module.append(nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size = i, padding = (i-1)//2, bias=True),
                nn.BatchNorm2d(oc),
                nn.Dropout2d(0.1),
            ))
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.deploy = False
        self.deploy_conv = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.deploy_conv(x))
        sum = None
        for i in range(self.cnt):
            o = self.list_module[i](x)
            if sum is None:
                sum = o
            else:
                sum = sum + o
        sum = sum / self.cnt
        o = self.act(sum)
        return o

    def reparameterize(self):
        # 合并卷积层的权重
        if self.deploy_conv is not None:
            return  # 如果已经部署了，就不再重新部署
        total_kernel = torch.zeros(self.list_module[0][0].weight.shape[0], self.list_module[0][0].weight.shape[1], 7, 7, device=self.list_module[0][0].weight.device)
        total_bias = torch.zeros(self.list_module[0][0].bias.shape, device=self.list_module[0][0].bias.device)

        for i, module in enumerate(self.list_module):
            # conv = module[0]
            # bn = module[1]
            # kernel, bias = self._fuse_bn_tensor(nn.Sequential(conv, bn))
            kernel, bias = self._fuse_bn_tensor(module)
            kernel = self._pad_to_7x7_tensor(kernel, i)
            total_kernel += kernel
            total_bias += bias
        total_kernel = total_kernel / self.cnt
        total_bias = total_bias / self.cnt
        # 创建一个新的7x7卷积层
        self.deploy_conv = nn.Conv2d(self.list_module[0][0].in_channels, self.list_module[0][0].out_channels,
                                     kernel_size=7, stride=1, padding=3, bias=True)
        self.deploy_conv.weight.data = total_kernel
        self.deploy_conv.bias.data = total_bias
        # self.deploy_conv.weight = torch.nn.Parameter(total_kernel)
        # self.deploy_conv.bias = torch.nn.Parameter(total_bias)
        # self.deploy_conv.weight.data = total_kernel / len(self.list_module)
        # self.deploy_conv.bias.data = total_bias / len(self.list_module)
        self.deploy = True

    # def _fuse_bn_tensor(self, module):
    #     if isinstance(module, nn.Sequential):
    #         kernel = module[0].weight  # 卷积层的权重。
    #         bias = module[0].bias  # 卷积层的偏置。
    #         running_mean = module[1].running_mean  # 批量归一化层的运行均值。
    #         running_var = module[1].running_var  # 批量归一化层的运行方差。
    #         gamma = module[1].weight  # 批量归一化层的缩放因子。
    #         beta = module[1].bias  # 批量归一化层的偏移量。
    #         eps = module[1].eps  # 批量归一化层的epsilon值，用于数值稳定性。
    #     # 计算标准差（std）
    #     std = (running_var + eps).sqrt()
    #     # 计算t = gamma / std，将gamma扩展为与卷积核相同的形状
    #     t = (gamma / std).reshape(-1, 1, 1, 1)
    #     # 返回融合后的卷积核和偏置
    #     return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, module):
        if isinstance(module, nn.Sequential):
            conv = module[0]
            bn = module[1]
            kernel = conv.weight  # 卷积层的权重。
            bias = conv.bias  # 卷积层的偏置。
            running_mean = bn.running_mean  # 批量归一化层的运行均值。
            running_var = bn.running_var  # 批量归一化层的运行方差。
            gamma = bn.weight  # 批量归一化层的缩放因子。
            beta = bn.bias  # 批量归一化层的偏移量。
            eps = bn.eps  # 批量归一化层的epsilon值，用于数值稳定性。

            # 计算标准差（std）
            std = (running_var + eps).sqrt()

            # 计算t = gamma / std，将gamma扩展为与卷积核相同的形状
            t = (gamma / std).reshape(-1, 1, 1, 1)

            # 计算融合后的卷积核
            fused_kernel = kernel * t

            # 计算融合后的偏置
            fused_bias = bias * gamma / std + beta - running_mean * gamma / std

            return fused_kernel, fused_bias

    def _pad_to_7x7_tensor(self, kernel, idx):
        if kernel is None:
            return torch.zeros_like(kernel)
        if idx == 0:  # 1x1
            return torch.nn.functional.pad(kernel, [3, 3, 3, 3])
        elif idx == 1:  # 3x3
            return torch.nn.functional.pad(kernel, [2, 2, 2, 2])
        elif idx == 2:  # 5x5
            return torch.nn.functional.pad(kernel, [1, 1, 1, 1])
        return kernel  # 7x7 kernel already in correct shape


class SimpleRep(nn.Module):
    def __init__(self,  param_list):
        super(SimpleRep, self).__init__()
        ch, layer, k = param_list
        self.head = BlockRep(1, ch, k)  # 首层是否使用rep

        ch, layer, k = int(ch), int(layer), int(k)
        layer_list = []
        for _ in range(layer):
            layer_list.append(BlockRep(ch, ch, k))
        self.body = nn.Sequential(*layer_list)
        self.tail = nn.Conv2d(ch, 1, kernel_size = 1, padding=0, bias=True)
        self.apply(InitWeights_He)
        

    def forward(self, x):
        o = self.head(x)
        o = self.body(o)
        o = self.tail(o)

        return o

    def reparameterize(self):
        self.head.reparameterize()
        for block in self.body:
            block.reparameterize()
