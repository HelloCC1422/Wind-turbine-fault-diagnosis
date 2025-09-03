# -*- encoding: utf-8 -*-
'''
  @Author   : changchen
  @Date     : 2024/12/25 15:26
  @PROJECT  : Programb
  @File     : DBFDformer.py    
  @Describe :
'''

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class Conv_block(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=None,):
        super(Conv_block, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size
        padding = padding or kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=output_size,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding
                               )
        self.bn = nn.BatchNorm1d(output_size)
        self.act1 = nn.PReLU()
        self.line = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act1(x)
        # x = self.line(x)
        # x = self.bn(x)
        # x = self.act1(x)
        return x

class Transformer_F(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Transformer_F, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv1 = Conv_block(1, 1)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size,  # 输入维度
                                                        nhead=1)  # 注意力头数
        # 构建Transformer编码器，参数包括编码层和层数
        self.encoder = nn.TransformerEncoder(self.encoder_layer,  # 编码层
                                             num_layers=1)  # 层数

        self.classifier = nn.Linear(self.input_size, self.num_classes)  # 输入维度 # 输出维度

    def forward(self, x):
        x = x.unsqueeze(1)  # 输入数据大小为（batch_size, in_channels, signal_len）
        # x = self.conv1(x)
        x = self.encoder(x)
        x = x.squeeze(1)
        x = self.classifier(x)

        return x

class Transformer_H(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Transformer_H, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # 构建Transformer编码层，参数包括输入维度、注意力头数
        # 其中d_model要和模型输入维度相同
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size,  # 输入维度
                                                        nhead=1)  # 注意力头数
        # 构建Transformer编码器，参数包括编码层和层数
        self.encoder = nn.TransformerEncoder(self.encoder_layer,  # 编码层
                                             num_layers=1)  # 层数

        # 构建线性层，参数包括输入维度和输出维度（num_classes）
        self.classifier = nn.Linear(self.input_size,  # 输入维度
                                    self.num_classes)  # 输出维度

    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        x = x.unsqueeze(1)
        # 将输入数据流经 Transformer 编码器进行特征提取
        x = self.encoder(x)
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        x = x.squeeze(1)
        # 将编码器的输出传入全连接层，获得最终输出结果
        x = self.classifier(x)
        return x
                             

# 创建DBFDformer模型
class DBFDformer(nn.Module):

    def __init__(self, input_size, num_classes):
        super(DBFDformer, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # 构建Transformer编码层，参数包括输入维度、注意力头数
        # 其中d_model要和模型输入维度相同
        self.H_former = Transformer_H(self.input_size, num_classes=2)
        self.F_former = Transformer_F(self.input_size, self.num_classes)

        self.dropout = nn.Dropout(1)  # 添加一个dropout层

    def forward(self, x):
        x1 = self.H_former(x)
        out1 = x1.argmax(dim=1)
        print("out1:", out1)
        if out1 == 1:
            x2 = self.F_former(x)
            out2 = x2.argmax(dim=1)
            return out2
        else:
            return out1
