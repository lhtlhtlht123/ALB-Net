import torch
import torch.nn as nn
from torchviz import make_dot

from models.utils import InitWeights_He


class Block(nn.Module):
    def __init__(self, ic, oc, k):
        super(Block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=k, padding=(k - 1) // 2, bias=True),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Simple(nn.Module):
    def __init__(self, param_list):
        super(Simple, self).__init__()
        ch, layer, k = param_list
        self.head = Block(1, ch, k)

        ch, layer, k = int(ch), int(layer), int(k)
        layer_list = []
        for _ in range(layer):
            layer_list.append(Block(ch, ch, k))
        self.body = nn.Sequential(*layer_list)
        self.tail = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He)

    def forward(self, x):
        o = self.head(x)
        o = self.body(o)
        o = self.tail(o)
        return o


# 初始化模型
model = Simple((16, 4, 3))  # 示例参数
x = torch.randn(1, 1, 256, 256)  # 示例输入

# 获取模型的计算图
y = model(x)
dot = make_dot(y.mean(), params=dict(model.named_parameters()))

# 渲染图像并保存
dot.render('model_graph', format='png')