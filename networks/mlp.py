import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_net import BaseNet


class MLP(BaseNet):
    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)

class MLP_Discriminator_S(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=64, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.num_class=2

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=False) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, self.num_class, bias=False)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return F.tanh(self.fc2(self.code(x)))

# class MLP_Autoencoder(nn.Module):
#     def __init__(self, x_dim, h_dims=[128, 64, 32], rep_dim=16, bias=False):
#         super(MLP_Autoencoder, self).__init__()
        
#         # Encoder: progressively reduces the dimensions
#         self.encoder = nn.Sequential(
#             nn.Linear(x_dim, h_dims[0]),
#             nn.ReLU(True),
#             nn.Linear(h_dims[0], h_dims[1]),
#             nn.ReLU(True),
#             nn.Linear(h_dims[1], h_dims[2]),
#             nn.ReLU(True),
#             nn.Linear(h_dims[2], rep_dim)
#         )
        
#         # Decoder: progressively reconstructs the input
#         self.decoder = nn.Sequential(
#             nn.Linear(rep_dim, h_dims[2]),
#             nn.ReLU(True),
#             nn.Linear(h_dims[2], h_dims[1]),
#             nn.ReLU(True),
#             nn.Linear(h_dims[1], h_dims[0]),
#             nn.ReLU(True),
#             nn.Linear(h_dims[0], x_dim),
#             nn.Sigmoid()  # Use Sigmoid to normalize output between 0 and 1
#         )
    
#     def forward(self, x, get_latent=False):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         if get_latent==True:
#             return decoded, encoded
#         else:
#             return decoded

class MLP_Autoencoder(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x, get_latent=False):
        latent = self.encoder(x)
        out = self.decoder(latent)
        if get_latent==True:
            return out, latent
        else:
            return out

class MLP_Decoder(BaseNet):

    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.ReLU()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)

class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))

class MLP_Classifier(BaseNet):

    def __init__(self, x_dim=32, h_dims=[128, 64], rep_dim=64, out_dim=2, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.out_dim = out_dim

        # 定义隐藏层
        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)

        # 定义特征提取层（代码层）
        self.code = Linear_BN_leakyReLU(h_dims[-1], self.rep_dim, bias=bias)

        # 定义分类层
        self.fc = nn.Linear(self.rep_dim, self.out_dim, bias=bias)
        # Sigmoid 函数（可选，用于二分类的输出）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, get_latent=False):
        x = x.view(int(x.size(0)), -1)  # 展平输入

        # 通过隐藏层
        for layer in self.hidden:
            x = layer(x)

        # 提取中间特征（latent）
        latent = self.code(x)

        # 输出分类结果
        out = self.fc(latent)

        # 根据参数决定是否返回 latent 特征
        if get_latent == False:
            return out  # 返回分类输出
        else:
            return self.sigmoid(out), latent  # 返回激活后的输出和 latent 特征

class MLP_Similiar(BaseNet):
    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=64, out_dim=2, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.out_dim = out_dim

        # 定义第一组隐藏层
        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=False) for i in range(1, len(neurons))]
        
        # 定义第二组隐藏层
        layers2 = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=False) for i in range(1, len(neurons))]

        # 分别存储两组隐藏层
        self.hidden = nn.ModuleList(layers)
        self.hidden2 = nn.ModuleList(layers2)

        # 定义特征提取层（代码层）
        self.code = Linear_BN_leakyReLU(h_dims[-1], self.rep_dim, bias=False)

        # 定义分类层，将两组隐藏层的输出连接
        self.fc2 = nn.Linear(self.rep_dim * 2, self.rep_dim, bias=False)
        self.fc = nn.Linear(self.rep_dim * 2, self.out_dim, bias=False)

        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # 展平输入
        x1 = x1.view(int(x1.size(0)), -1)
        #x2 = x2.view(int(x2.size(0)), -1)

        # 通过第一组隐藏层处理 x1
        for layer in self.hidden:
            x1 = layer(x1)

        # 通过第二组隐藏层处理 x2
        for layer in self.hidden2:
            x2 = layer(x2)

        # 提取 x1 和 x2 的中间特征
        # latent1 = self.code(x1)
        # latent2 = self.code(x2)

        # 将 latent1 和 latent2 连接，作为输入到分类层
        combined_latent = torch.cat([x1, x2], dim=1)

        # 输出分类结果
        out = self.fc(combined_latent)

        return F.leaky_relu(out)
