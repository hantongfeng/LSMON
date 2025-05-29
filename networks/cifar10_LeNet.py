import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc3 = nn.Linear(128, self.rep_dim, bias=False)
        self.bn1d2 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        #x = self.bn1d2(self.fc3(x))
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x



class CIFAR10_Discriminator_L(BaseNet):

    def __init__(self, x_dim=128, h_dims=[128,256,128], rep_dim=128,out_dim = 2, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.out_dim = out_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = Linear_BN_leakyReLU(h_dims[-1], self.rep_dim, bias=bias)
        self.fc  = nn.Linear(self.rep_dim,self.out_dim,bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,get_latent=False):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        latent = self.code(x)
        out = self.fc(latent)
        if get_latent==False:
            return out
        else:
            return self.sigmoid(out), latent


class CIFAR10_Similiar(BaseNet): 
    def __init__(self, x_dim=128, h_dims=[128,256,128], rep_dim=128, out_dim = 2, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.out_dim = out_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        layers2 = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(layers)
        self.hidden2 = nn.ModuleList(layers2)
        self.code = Linear_BN_leakyReLU(h_dims[-1], self.rep_dim, bias=bias)
        self.fc  = nn.Linear(self.rep_dim*2,self.rep_dim,bias=bias)
        self.fc2  = nn.Linear(self.rep_dim,32,bias=bias)
        self.fc3  = nn.Linear(32,self.out_dim,bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Linear(self.rep_dim*2, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_dim),
            #nn.Sigmoid()
        )

    def forward(self, x,x2, get_latent=False): # x2,
        x = x.view(int(x.size(0)), -1) # torch.Size([89, 64])
        for layer in self.hidden:
            x = layer(x)
        for layer2 in self.hidden2:
            x2=layer2(x2)
        latent=torch.cat([x, x2], dim=1)
        ###########################
        out = self.fc(latent)
        out = self.fc2(out)
        out = self.fc3(out)
        ###########################
        #out = self.classifier(latent)

        if get_latent==False:
            return out
        else:
            return self.sigmoid(out), latent
        
class CIFAR10_Discriminator_S(BaseNet):

    def __init__(self, rep_dim=128,num_class=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.num_class = num_class
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, self.num_class, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, get_latent=False):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        latent = F.leaky_relu(self.fc1(x))
        out = self.fc2(latent)
        if get_latent==False:
            return out
        else:
            return out, latent



class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x,get_latent=False):
        latent = self.encoder(x)
        out = self.decoder(latent)
        if get_latent==True:
            return out, latent
        else:
            return out


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
