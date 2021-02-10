import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def photonic_sigmoid(x, cutoff=2):
    A1 = 0.060
    A2 = 1.005
    x0 = 0.145
    d = 0.033
    x = x - x0
    x[x > cutoff] = 1
    y = A2 + (A1 - A2) / (1 + torch.exp(x / d))
    return y

#TODO: had to manually add channels = 3 because of loader changes. Fix it
class Boftrainer(nn.Module):
    def __init__(self,arch,centers,sigma,bofnumber, activation):
        super().__init__()
        self.arch = arch
        self.clusternumber = centers.size(0)
        self.bofnumber = bofnumber
        channels = 3
        if arch == 1:
            self.conv1 = nn.Conv2d(channels, 8, kernel_size = 3, stride = 2).to(device)
            self.conv2 = nn.Conv2d(8, 8, kernel_size = 3, stride = 1).to(device)
            self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 1).to(device)
            self.conv3 = nn.Conv2d(8, 6, kernel_size = 3, stride = 1).to(device)
            self.conv4 = nn.Conv2d(6, 6, kernel_size = 3, stride = 1).to(device)
            
        elif arch == 2:
            self.conv1 = nn.Conv2d(channels, 16, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
            self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2).to(device)
            self.conv3 = nn.Conv2d(16, 24, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.conv4 = nn.Conv2d(24, 16, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
        
        elif arch == 3:
            self.conv1 = nn.Conv2d(channels, 9, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
            self.conv2 = nn.Conv2d(9, 9, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2).to(device)
            self.conv3 = nn.Conv2d(9, 10, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.conv4 = nn.Conv2d(10, 9, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
        
        self.l1 = nn.Linear(self.clusternumber, 20, bias = True).to(device)
        self.l2 = nn.Linear(20, 10, bias = True).to(device)
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.codebook = nn.Parameter(centers, requires_grad=True)
        if activation == 'relu':
            self.activations = lambda r : torch.relu(r)
        if activation == 'sin':
            self.activations = lambda r : torch.sin(r) ** 2
        if activation == 'tanh':
            self.activations = lambda r : torch.tanh(r)
        if activation == 'photosig':
            self.activations = lambda r : photonic_sigmoid(r)

    def forward(self,x):
        if self.bofnumber >= 1:
            x = self.activations(self.conv1(x.to(device))).to(device)
            if self.bofnumber >=2:
                x = self.activations(self.conv2(x.to(device))).to(device)
                if self.bofnumber >= 3:
                    if self.arch == 2 or self.arch == 3:
                        x = self.pool2(x)
                    elif self.arch == 1:
                        x = self.pool1(x)
                    x = self.activations(self.conv3(x.to(device))).to(device)
                    if self.bofnumber == 4:
                        x = self.activations(self.conv4(x.to(device))).to(device)

        x = torch.flatten(x, start_dim = 2, end_dim=3).to(device)#receives output of conv layer OR original input
        x = x.transpose(1,2)
        x = x.unsqueeze(1)

        #NOTE: x is detached here //detach was deprecated 07-02-21 as it did not seem to slow down / speed up codebook training
        x = torch.exp(-(x.to(device) - self.codebook.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma.unsqueeze(0).unsqueeze(2).to(device))
        x = F.normalize(x, 1, dim = 1).to(device)
        x = torch.mean(x, dim = 2)

        x = torch.relu(self.l1(x))
        x = (self.l2(x))
        return x