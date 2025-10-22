import torch
import torch.nn as nn
import torch.nn.functional as F

from .fadc_block import AdaptiveDilatedConv
from .utils import InitWeights_He_Resnet

class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

class SE_ASPP(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.senet=SE_Block(in_planes=dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             #加入通道注意力机制
        # print('seaspp1:',seaspp1.shape)
        se_feature_cat=seaspp1*feature_cat
        result = self.conv_cat(se_feature_cat)
        # print('result:',result.shape)
        return result



class BlockRepRes(nn.Module):
    def __init__(self, ic, oc, k):
        super(BlockRepRes, self).__init__()
        self.list_module = nn.ModuleList()
        self.cnt = 0
        self.kernel_sizes = []  # 记录卷积核大小
        self.alphas = nn.ParameterList()  # 动态调整比例的参数
        for i in range(1, k + 1, 2):
            self.cnt += 1
            self.kernel_sizes.append(i)  #
            # 添加动态调整比例的参数，初始化为均匀分布
            alpha = nn.Parameter(torch.ones(1) / self.cnt)
            self.alphas.append(alpha)
            self.list_module.append(nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=i, padding=(i - 1) // 2, bias=True),
                nn.BatchNorm2d(oc),
                nn.Dropout2d(0.1),
            ))
        self.skip_connection = nn.Conv2d(ic, oc, kernel_size=1, bias=False) if ic == oc else None
        # self.skip_connection = nn.Conv2d(ic, oc, kernel_size=1, bias=True)
        # self.skip_connection = None
        # self.identity = nn.BatchNorm2d(oc) if ic == oc else None
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.deploy = False
        self.deploy_conv = None
        # self.weight_gen = nn.Sequential(
        #     nn.Conv2d(ic, self.cnt * 2, kernel_size=1, bias=True),
        #     nn.Softmax(dim=1)
        # )  #

    def forward(self, x):
        if self.deploy:
            sum = self.deploy_conv(x)
            if self.skip_connection is None:
                sum = sum + x
            return self.act(sum)
        # 下面是原来的
        # sum = None
        # for i in range(self.cnt):
        #     o = self.list_module[i](x)
        #     if sum is None:
        #         sum = o
        #     else:
        #         sum = sum + o
        # sum = sum / self.cnt
        # 动态分配权重
        # weights = self.weight_gen(x)  # shape: [batch_size, cnt*2, H, W]
        # low_weights = weights[:, :self.cnt]
        # high_weights = weights[:, self.cnt:]
        #
        # sum_low = None
        # sum_high = None
        # for i in range(self.cnt):
        #     o = self.list_module[i](x)
        #     if sum_low is None:
        #         sum_low = o * low_weights[:, i:i+1]
        #         sum_high = o * high_weights[:, i:i+1]
        #     else:
        #         sum_low = sum_low + o * low_weights[:, i:i+1]
        #         sum_high = sum_high + o * high_weights[:, i:i+1]
        # sum = sum_low + sum_high
        # sum = sum / self.cnt
        # 使用softmax确保比例和为1
        alphas = torch.softmax(torch.cat([alpha for alpha in self.alphas]), dim=0)
        sum = None
        for i in range(self.cnt):
            o = self.list_module[i](x)
            if sum is None:
                sum = o * alphas[i]
            else:
                sum = sum + o * alphas[i]
        sum = sum  # 不需要除以cnt，因为已经通过softmax归一化
        # Apply skip connection
        if self.skip_connection is not None:
            x = self.skip_connection(x)
        sum = sum + x

        o = self.act(sum)
        return o

    def reparameterize(self):
        # 合并卷积层的权重
        if self.deploy_conv is not None:
            return  # 如果已经部署了，就不再重新部署
        total_kernel = torch.zeros(self.list_module[0][0].weight.shape[0], self.list_module[0][0].weight.shape[1], 7, 7, device=self.list_module[0][0].weight.device)
        total_bias = torch.zeros(self.list_module[0][0].bias.shape, device=self.list_module[0][0].bias.device)
        # 使用softmax确保比例和为1
        alphas = torch.softmax(torch.cat([alpha for alpha in self.alphas]), dim=0)

        for i, module in enumerate(self.list_module):
            # conv = module[0]
            # bn = module[1]
            # kernel, bias = self._fuse_bn_tensor(nn.Sequential(conv, bn))
            kernel, bias = self._fuse_bn_tensor(module)
            kernel = self._pad_to_7x7_tensor(kernel, i)
            total_kernel += kernel * alphas[i]
            total_bias += bias * alphas[i]
            # total_kernel += kernel
            # total_bias += bias

        # total_kernel = total_kernel / self.cnt
        # total_bias = total_bias / self.cnt
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
        # self.deploy_conv.weight.data = total_kernel / (len(self.list_module) + int(self.skip_connection is not None))
        # self.deploy_conv.bias.data = total_bias / (len(self.list_module) + int(self.skip_connection is not None))
        self.deploy = True

        # 打印alpha参数的值
        print("Alpha parameters after softmax:", alphas)
        for j, alpha in enumerate(alphas):
            print(f"Conv layer {j + 1} with kernel size {self.kernel_sizes[j]} contribution: {alpha.item():.4f}")


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

    def _pad_to_7x7_tensor(self, kernel, idx):
        if kernel is None:
            return torch.zeros_like(kernel)
        if idx == 0:  # 1x1
            return torch.nn.functional.pad(kernel, [3, 3, 3, 3])
        elif idx == 1:  # 3x3
            return torch.nn.functional.pad(kernel, [2, 2, 2, 2])
        elif idx == 2:  # 5x5
            return torch.nn.functional.pad(kernel, [1, 1, 1, 1])
        elif idx == -1:  # Skip connection
            return torch.nn.functional.pad(kernel, [3, 3, 3, 3])
        return kernel  # 7x7 kernel already in correct shape

class SimpleRepRes(nn.Module):
    def __init__(self, param_list):
        super(SimpleRepRes, self).__init__()
        ch, layer, k = param_list
        self.head = BlockRepRes(1, ch, k)

        ch, layer, k = int(ch), int(layer), int(k)
        layer_list = []
        for _ in range(layer):
            layer_list.append(BlockRepRes(ch, ch, k))
        self.body = nn.Sequential(*layer_list)

        # 添加 SE_ASPP 模块
        # self.se_aspp = SE_ASPP(dim_in=ch, dim_out=ch, rate=1, bn_mom=0.1)

        # 替换 SE_ASPP 为 AdaptiveDilatedConv
        # self.adaptive_dilated_conv = AdaptiveDilatedConv(in_channels=ch, out_channels=ch, kernel_size=3)

        self.tail = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He_Resnet)

    def forward(self, x):
        o = self.head(x)
        o = self.body(o)
        # o = self.se_aspp(o)  # 在 tail 之前添加 SE_ASPP 模块
        # o = self.adaptive_dilated_conv(o)  # 使用 AdaptiveDilatedConv 替代 SE_ASPP
        o = self.tail(o)
        return o

    def reparameterize(self):
        self.head.reparameterize()
        for block in self.body:
            block.reparameterize()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
