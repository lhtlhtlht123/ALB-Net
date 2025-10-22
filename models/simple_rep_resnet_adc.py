import torch
import torch.nn as nn
import torch.nn.functional as F

from .fadc_block import AdaptiveDilatedConv
from .utils import InitWeights_He_Resnet

class BlockRepRes(nn.Module):
    def __init__(self, ic, oc, k, branch="small"):
        super(BlockRepRes, self).__init__()
        self.list_module = nn.ModuleList()
        self.cnt = 0
        self.kernel_sizes = []  # 记录卷积核大小
        self.branch = branch  # "small" or "large"

        # 根据分支选择卷积核大小
        if branch == "small":
            kernel_sizes = [1, 3]
        elif branch == "large":
            kernel_sizes = [5, 7]
        else:
            raise ValueError("branch must be 'small' or 'large'")

        for i in kernel_sizes:
            self.cnt += 1
            self.kernel_sizes.append(i)
            self.list_module.append(nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=i, padding=(i - 1) // 2, bias=True),
                nn.BatchNorm2d(oc),
                nn.Dropout2d(0.1),
            ))
        self.skip_connection = nn.Conv2d(ic, oc, kernel_size=1, bias=False) if ic == oc else None
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.deploy = False
        self.deploy_conv = None

    def forward(self, x):
        if self.deploy:
            sum = self.deploy_conv(x)
            if self.skip_connection is None:
                sum = sum + x
            return self.act(sum)
        sum = None
        for i in range(self.cnt):
            o = self.list_module[i](x)
            if sum is None:
                sum = o
            else:
                sum = sum + o
        sum = sum / self.cnt  # 平均池化
        # Apply skip connection
        if self.skip_connection is not None:
            x = self.skip_connection(x)
        sum = sum + x

        o = self.act(sum)
        return o

    # def reparameterize(self):
    #     if self.deploy_conv is not None:
    #         return  # 如果已经部署了，就不再重新部署
    #
    #     # 合并卷积层的权重
    #     total_kernel = None
    #     total_bias = None
    #
    #     for i, module in enumerate(self.list_module):
    #         kernel, bias = self._fuse_bn_tensor(module)
    #         if total_kernel is None:
    #             total_kernel = kernel
    #             total_bias = bias
    #         else:
    #             total_kernel += kernel
    #             total_bias += bias
    #
    #     if self.skip_connection is not None:
    #         skip_kernel, skip_bias = self._fuse_bn_tensor(self.skip_connection)
    #         total_kernel += skip_kernel
    #         total_bias += skip_bias
    #
    #     # 创建一个新的卷积层
    #     if self.branch == "small":
    #         kernel_size = 3
    #         padding = 1
    #         # 将1x1卷积核填充为3x3
    #         if self.kernel_sizes[0] == 1:
    #             total_kernel = torch.nn.functional.pad(total_kernel, [1, 1, 1, 1])
    #     else:
    #         kernel_size = 7
    #         padding = 3
    #         # 将5x5卷积核填充为7x7
    #         if self.kernel_sizes[0] == 5:
    #             total_kernel = torch.nn.functional.pad(total_kernel, [1, 1, 1, 1])
    #
    #     self.deploy_conv = nn.Conv2d(
    #         self.list_module[0][0].in_channels,
    #         self.list_module[0][0].out_channels,
    #         kernel_size=kernel_size,
    #         stride=1,
    #         padding=padding,
    #         bias=True
    #     )
    #     self.deploy_conv.weight.data = total_kernel / self.cnt
    #     self.deploy_conv.bias.data = total_bias / self.cnt
    #     self.deploy = True

    def reparameterize(self):
        # 合并卷积层的权重
        if self.deploy_conv is not None:
            return  # 如果已经部署了，就不再重新部署
        # 使用softmax确保比例和为1
        # alphas = torch.softmax(torch.cat([alpha for alpha in self.alphas]), dim=0)
        if self.branch == "large":
            total_kernel = torch.zeros(self.list_module[0][0].weight.shape[0], self.list_module[0][0].weight.shape[1],
                                       7, 7, device=self.list_module[0][0].weight.device)
            total_bias = torch.zeros(self.list_module[0][0].bias.shape, device=self.list_module[0][0].bias.device)
            for i, module in enumerate(self.list_module):
                # conv = module[0]
                # bn = module[1]
                # kernel, bias = self._fuse_bn_tensor(nn.Sequential(conv, bn))
                kernel, bias = self._fuse_bn_tensor(module)
                kernel = self._pad_to_7x7_tensor(kernel, i)
                # total_kernel += kernel * alphas[i]
                # total_bias += bias * alphas[i]
                total_kernel += kernel
                total_bias += bias

            total_kernel = total_kernel / self.cnt
            total_bias = total_bias / self.cnt
            if self.skip_connection is not None:
                skip_kernel, skip_bias = self._fuse_bn_tensor(self.skip_connection)
                # skip_kernel = self.skip_connection.weight
                # skip_bias = self.skip_connection.bias
                skip_kernel = self._pad_to_7x7_tensor(skip_kernel, -1)  # Use -1 as an indicator for skip connection
                total_kernel += skip_kernel
                total_bias += skip_bias

            # 创建一个新的7x7卷积层
            self.deploy_conv = nn.Conv2d(self.list_module[0][0].in_channels, self.list_module[0][0].out_channels,
                                         kernel_size=7, stride=1, padding=3, bias=True)
            self.deploy_conv.weight.data = total_kernel
            self.deploy_conv.bias.data = total_bias
        if self.branch == "small":
            total_kernel = torch.zeros(self.list_module[0][0].weight.shape[0], self.list_module[0][0].weight.shape[1],
                                       3, 3, device=self.list_module[0][0].weight.device)
            total_bias = torch.zeros(self.list_module[0][0].bias.shape, device=self.list_module[0][0].bias.device)
            for i, module in enumerate(self.list_module):
                # conv = module[0]
                # bn = module[1]
                # kernel, bias = self._fuse_bn_tensor(nn.Sequential(conv, bn))
                kernel, bias = self._fuse_bn_tensor(module)
                kernel = self._pad_to_3x3_tensor(kernel, i)
                # total_kernel += kernel * alphas[i]
                # total_bias += bias * alphas[i]
                total_kernel += kernel
                total_bias += bias

            total_kernel = total_kernel / self.cnt
            total_bias = total_bias / self.cnt
            if self.skip_connection is not None:
                skip_kernel, skip_bias = self._fuse_bn_tensor(self.skip_connection)
                # skip_kernel = self.skip_connection.weight
                # skip_bias = self.skip_connection.bias
                skip_kernel = self._pad_to_3x3_tensor(skip_kernel, -1)  # Use -1 as an indicator for skip connection
                total_kernel += skip_kernel
                total_bias += skip_bias

            # 创建一个新的7x7卷积层
            self.deploy_conv = nn.Conv2d(self.list_module[0][0].in_channels, self.list_module[0][0].out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=True)
            self.deploy_conv.weight.data = total_kernel
            self.deploy_conv.bias.data = total_bias
        # self.deploy_conv.weight.data = total_kernel / (len(self.list_module) + int(self.skip_connection is not None))
        # self.deploy_conv.bias.data = total_bias / (len(self.list_module) + int(self.skip_connection is not None))
        self.deploy = True

        # # 打印alpha参数的值
        # print("Alpha parameters after softmax:", alphas)
        # for j, alpha in enumerate(alphas):
        #     print(f"Conv layer {j + 1} with kernel size {self.kernel_sizes[j]} contribution: {alpha.item():.4f}")


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
        elif isinstance(module, nn.Conv2d):

            kernel = module.weight
            bias = module.bias if module.bias is not None else torch.zeros(module.out_channels, device=kernel.device)
            # running_mean = torch.zeros_like(bias)
            # running_var = torch.ones_like(bias)
            # gamma = torch.ones_like(bias)
            # beta = torch.zeros_like(bias)
            # eps = 1e-5
            return kernel, bias

            # 计算标准差（std）
        std = (running_var + eps).sqrt()

            # 计算t = gamma / std，将gamma扩展为与卷积核相同的形状
        t = (gamma / std).reshape(-1, 1, 1, 1)

            # 计算融合后的卷积核
        fused_kernel = kernel * t

            # 计算融合后的偏置
        fused_bias = bias * gamma / std + beta - running_mean * gamma / std

        return fused_kernel, fused_bias

    # def _pad_to_7x7_tensor(self, kernel, idx):
    #     if kernel is None:
    #         return torch.zeros_like(kernel)
    #     if idx == 0:  # 1x1
    #         return torch.nn.functional.pad(kernel, [3, 3, 3, 3])
    #     elif idx == 1:  # 3x3
    #         return torch.nn.functional.pad(kernel, [2, 2, 2, 2])
    #     elif idx == 2:  # 5x5
    #         return torch.nn.functional.pad(kernel, [1, 1, 1, 1])
    #     elif idx == -1:  # Skip connection
    #         return torch.nn.functional.pad(kernel, [3, 3, 3, 3])
    #     return kernel  # 7x7 kernel already in correct shape

    def _pad_to_7x7_tensor(self, kernel, idx):
        if kernel is None:
            return torch.zeros_like(kernel)
        if idx == 0:  # 5x5
            return torch.nn.functional.pad(kernel, [1, 1, 1, 1])
        elif idx == -1:  # Skip connection
            return torch.nn.functional.pad(kernel, [3, 3, 3, 3])
        return kernel  # 7x7 kernel already in correct shape

    def _pad_to_3x3_tensor(self, kernel, idx):
        if kernel is None:
            return torch.zeros_like(kernel)
        if idx == 0:  # 1x1
            return torch.nn.functional.pad(kernel, [1, 1, 1, 1])
        elif idx == -1:  # Skip connection
            return torch.nn.functional.pad(kernel, [1, 1, 1, 1])
        return kernel  # 7x7 kernel already in correct shape

class SimpleRepResAdc(nn.Module):
    def __init__(self, param_list):
        super(SimpleRepResAdc, self).__init__()
        ch, layer, k = param_list
        ch, layer, k = int(ch), int(layer), int(k)
        layer_small = layer // 2  # 小分支的层数
        layer_large = layer - layer_small  # 大分支的层数

        # 小分支的头部和主体
        self.small_head = BlockRepRes(1, ch, k, branch="small")
        small_list = []
        for _ in range(layer_small):
            small_list.append(BlockRepRes(ch, ch, k, branch="small"))
        self.small_body = nn.Sequential(*small_list)

        # 大分支的头部和主体
        self.large_head = BlockRepRes(1, ch, k, branch="large")
        large_list = []
        for _ in range(layer_large):
            large_list.append(BlockRepRes(ch, ch, k, branch="large"))
        self.large_body = nn.Sequential(*large_list)

        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 初始化为0.5

        self.tail = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # 小分支
        o_small = self.small_head(x)
        o_small = self.small_body(o_small)

        # 大分支
        o_large = self.large_head(x)
        o_large = self.large_body(o_large)

        # 动态调整两分支的权重
        o = self.alpha * o_small + (1 - self.alpha) * o_large
        o = self.tail(o)
        #只用小分支
        # o = self.tail(o_small)
        return o

    def reparameterize(self):
        self.small_head.reparameterize()
        for block in self.small_body:
            block.reparameterize()
        self.large_head.reparameterize()
        for block in self.large_body:
            block.reparameterize()
        # 打印alpha参数的值
        print(f"Alpha parameter value: {self.alpha.item():.4f}")

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
