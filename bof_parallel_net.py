import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import bof_trainer
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def calculate_bof_histogram(inputrep, codebook, kernel, sigma, number):
    '''
    input: output of a Conv layer in pytorch
    codebook: codebook of bof layer
    kernel: hyperbolic or rbf
    sigma: bandwiths
    number: 0 for 1st bof, 4th for last
    '''
    flattenedinp = torch.flatten(inputrep, start_dim = 2, end_dim=3).to(device)
    if kernel == 'hyperbolic':
        featvecs = torch.tanh(torch.matmul(flattenedinp.transpose(1,2),codebook.T))
        featvecs = featvecs.transpose(1,2)
        featvecs = 1 / (1 + torch.exp(-1. * featvecs *2 * 0.36 + 0.5*1))
    else:
        featvecs = flattenedinp.transpose(1,2)
        featvecs = featvecs.unsqueeze(1)
        codebook = codebook.unsqueeze(0).unsqueeze(2)
        featvecs = torch.exp(-(featvecs - codebook).abs().pow(2).sum(3) * sigma[number].unsqueeze(0).unsqueeze(2))#removed [None,:, None]
    featvecs = torch.clamp(featvecs, min=0.000001)
    featvecs = F.normalize(featvecs, 1, dim = 1)
    return torch.mean(featvecs, dim = 2)

def photonic_sigmoid(x, cutoff=2):
    A1 = 0.060
    A2 = 1.005
    x0 = 0.145
    d = 0.033
    x = x - x0
    x[x > cutoff] = 1
    y = A2 + (A1 - A2) / (1 + torch.exp(x / d))
    return y



class ConvBOFVGG(nn.Module):
    def __init__(self, center_initial, center_initial_y, center_train, center_train_y,  clusters, arch, quant_input, end_with_linear, activation, path, exp_number):
        super().__init__()
        self.arch = arch
        self.path = path
        self.experiment_number = exp_number
        self.center_initializer = center_initial.to(device)
        self.center_initializer_y = center_initial_y.to(device)
        self.center_train = center_train
        self.center_train_y = center_train_y
        self.imgsize = self.center_initializer.size(2)
        self.channels = self.center_initializer.size(1)

        if arch == 1:
            self.conv1 = nn.Conv2d(self.channels, 8, kernel_size = 3, stride = 2).to(device)
            self.conv2 = nn.Conv2d(8, 8, kernel_size = 3, stride = 1).to(device)
            self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 1).to(device)
            self.conv3 = nn.Conv2d(8, 6, kernel_size = 3, stride = 1).to(device)
            self.conv4 = nn.Conv2d(6, 6, kernel_size = 3, stride = 1).to(device)
            self.conv5 = nn.Conv2d(6, 10, kernel_size = int(self.imgsize / 4), stride = 1).to(device)
            self.sizeforinit1 = (self.center_initializer.size(0), int(self.imgsize**2 / 2) - 1,8)
            self.sizeforinit2 = (self.center_initializer.size(0), int(self.imgsize**2 / 2) - 3,8)
            self.sizeforinit3 = (self.center_initializer.size(0),int(self.imgsize**2 / 2) - 5, 6)
            self.sizeforinit4 = (self.center_initializer.size(0),int(self.imgsize**2 / 2) - 7, 6)
            
        elif arch == 2:
            self.conv1 = nn.Conv2d(self.channels, 16, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
            self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2).to(device)
            self.conv3 = nn.Conv2d(16, 24, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.conv4 = nn.Conv2d(24, 16, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
            self.pool2_2 = nn.MaxPool2d(kernel_size = 2, stride = 2).to(device)
            self.conv5 = nn.Conv2d(16, 10, kernel_size = int(self.imgsize / 4), stride = 1, bias = True).to(device)
            self.sizeforinit1 = (self.center_initializer.size(0), self.imgsize**2,16)
            self.sizeforinit2 = (self.center_initializer.size(0), self.imgsize**2,16)
            self.sizeforinit3 = (self.center_initializer.size(0),(int(self.imgsize/2))**2, 24)
            self.sizeforinit4 = (self.center_initializer.size(0),(int(self.imgsize/2))**2, 16)
        elif arch == 3:
            self.conv1 = nn.Conv2d(self.channels, 9, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
            self.conv2 = nn.Conv2d(9, 9, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2).to(device)
            self.conv3 = nn.Conv2d(9, 10, kernel_size = 3, stride = 1, padding = (1,1), bias = True).to(device)
            self.conv4 = nn.Conv2d(10, 9, kernel_size = 3, stride = 1, padding=(1,1), bias = True).to(device)
            self.pool2_2 = nn.MaxPool2d(kernel_size = 2, stride = 2).to(device)
            self.conv5 = nn.Conv2d(9, 10, kernel_size = int(self.imgsize / 4), stride = 1, bias = True).to(device)
            self.sizeforinit1 = (self.center_initializer.size(0), self.imgsize**2,9)
            self.sizeforinit2 = (self.center_initializer.size(0), self.imgsize**2,9)
            self.sizeforinit3 = (self.center_initializer.size(0),(int(self.imgsize/2))**2, 10)
            self.sizeforinit4 = (self.center_initializer.size(0),(int(self.imgsize/2))**2, 9)            

        self.quant_input = quant_input
        self.end_with_linear = end_with_linear
        self.softmax = nn.Softmax(dim=1)
        
        self.clusternumber = clusters
        self.start_bof_training = False
        if arch  <= 1:
            self.arch1 = True
        else:
            self.arch1 = False

        if end_with_linear:#If we want to replace the conv layer after 4 convs with a classifier
            self.nlib0 = nn.Linear(1024, 5).to(device)
            self.nlib1 = nn.Linear(5, 12).to(device)
            self.nlib2 = nn.Linear(12, 10).to(device)
        
        
        self.sigma0 = (torch.ones(size = (1, self.clusternumber)) * 0.90).to(device)
        self.sigma1 = (torch.ones(size = (1, self.clusternumber)) * 0.75).to(device)
        self.sigma2 = (torch.ones(size = (1, self.clusternumber)) * 0.65).to(device)
        self.sigma3 = (torch.ones(size = (1, self.clusternumber)) * 0.50).to(device)
        self.sigma4 = (torch.ones(size = (1, self.clusternumber)) * 0.40).to(device)
        self.sigma = torch.stack((self.sigma0, self.sigma1,self.sigma2,self.sigma3,self.sigma4)).squeeze(1)
        self.sigma.requires_grad=False
        #IF During kt we want to train sigma along with centers then we may set to true

        self.student_network = False
        
        self.activation = activation
        if self.activation == 'relu':
            self.activations = lambda r : torch.relu(r)
        if self.activation == 'sin':
            self.activations = lambda r : torch.sin(r) ** 2
        if self.activation == 'tanh':
            self.activations = lambda r : torch.tanh(r)
        if self.activation == 'celu':
            self.activations = lambda r : torch.celu(r)
        if self.activation == 'photosig':
            self.activations = lambda r : photonic_sigmoid(r)

    def prepare_centers(self, k_means_iterations = 500, train_iterations = 130, n_initializations = 1):
        '''
        This function calculates the centers of each bof layer
        Initializes codebook with K-means after passing each instance of center_initializer through the network
        then gathering every feature vector and clustering the (total_feature_vectors) x (filters) ''dataset''
        Then it trains a network with input the ceter_initializer instance and output the corresponding labels
        A total of 5 networks is trained for each bof layer (placing the corresponding bof layer after the corresponding conv layer)
        Both codebook and sigma change during training
        '''
        print('Initializing centers ...')
        KMC = KMeans(n_clusters=self.clusternumber, max_iter = k_means_iterations, n_init = n_initializations)
        #if self.quant_input:
        #    original = self.center_initializer.to(device).detach()
        #    original = torch.flatten(original, start_dim = 2, end_dim = 3)
        #    original = torch.reshape(original.transpose(1,2), (self.center_initializer.size(0), self.imgsize**2,self.channels))
        #    clusters = np.array((torch.reshape(original, (self.center_initializer.size(0)*self.imgsize**2, self.channels)).cpu()))
        #    clusters = torch.tensor(KMC.fit(clusters).cluster_centers_, requires_grad = True)
        #    self.codebook0 = clusters

        aa1init = self.activations(self.conv1(self.center_initializer.to(device)).to(device)).detach()
        aa1 = torch.flatten(aa1init, start_dim = 2, end_dim=3)
        aa1 = torch.reshape(aa1.transpose(1,2), (self.sizeforinit1))
        clusters = np.array((torch.reshape(aa1, (self.sizeforinit1[0] * self.sizeforinit1[1], self.sizeforinit1[2])).cpu()))
        clusters = torch.tensor(KMC.fit(clusters).cluster_centers_, requires_grad = True)
        self.codebook1 = clusters
        
        aa2init = self.activations(self.conv2(aa1init.to(device)).to(device)).detach()
        aa2 = torch.flatten(aa2init, start_dim = 2, end_dim=3)
        aa2 = torch.reshape(aa2.transpose(1,2), (self.sizeforinit2))
        clusters = np.array((torch.reshape(aa2, (self.sizeforinit2[0] * self.sizeforinit2[1], self.sizeforinit2[2])).cpu()))
        clusters = torch.tensor(KMC.fit(clusters).cluster_centers_, requires_grad = True)
        self.codebook2 = clusters

        aa3init = self.activations(self.conv3(self.pool2(aa2init.to(device))).to(device)).detach()
        aa3 = torch.flatten(aa3init, start_dim = 2, end_dim=3)
        aa3 = torch.reshape(aa3.transpose(1,2), (self.sizeforinit3))
        clusters = np.array((torch.reshape(aa3, (self.sizeforinit3[0] * self.sizeforinit3[1], self.sizeforinit3[2])).cpu()))
        clusters = torch.tensor(KMC.fit(clusters).cluster_centers_, requires_grad = True)
        self.codebook3 = clusters

        aa4init = self.activations(self.conv4(aa3init).to(device)).detach()
        aa4 = torch.flatten(aa4init, start_dim = 2, end_dim=3)
        aa4 = torch.reshape(aa4.transpose(1,2), (self.sizeforinit4))
        clusters = np.array((torch.reshape(aa4, (self.sizeforinit4[0] * self.sizeforinit4[1], self.sizeforinit4[2])).cpu()))
        clusters = torch.tensor(KMC.fit(clusters).cluster_centers_, requires_grad = True)
        self.codebook4 = clusters

        #Call the function train_bof_centers_with_ce to train the extracted cbs with ce loss and trained sigmas
        #self.codebook0, self.sigma[0] = self.train_bof_centers_with_ce(self.codebook0, self.sigma[0], 0, train_iterations, self.center_initializer, self.center_initializer_y)
        self.codebook1, self.sigma[1] = self.train_bof_centers_with_ce(self.codebook1, self.sigma[1], 1, train_iterations, self.center_train)
        print('after', (self.codebook1[4]))
        self.codebook2, self.sigma[2] = self.train_bof_centers_with_ce(self.codebook2, self.sigma[2], 2, train_iterations, self.center_train)
        print('after', (self.codebook2[4]))
        self.codebook3, self.sigma[3] = self.train_bof_centers_with_ce(self.codebook3, self.sigma[3], 3, train_iterations, self.center_train)
        print('after', (self.codebook3[4]))
        self.codebook4, self.sigma[4] = self.train_bof_centers_with_ce(self.codebook4, self.sigma[4], 4, train_iterations, self.center_train)

        #If we want to train centers for student
        if self.student_network:
            self.codebook1.requires_grad = True
            self.codebook2.requires_grad = True
            self.codebook3.requires_grad = True
            self.codebook4.requires_grad = True


        #IF we want a trainable codebook during kt then we may set it as nn.Parameter here
    def train_bof_centers_with_ce(self, codebook, sigma, bof_number, iterations, train_loader):
        '''
        This function is used by prepare_centers to train the networks
        returns trained codebook and trained sigma
        iterations: How many times should data (center_initializer) go through the network
        '''
        print('before', (codebook[4]))
        model = bof_trainer.Boftrainer(self.arch, codebook, sigma, bof_number, self.activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.08)
        for epoch in range(iterations):
            for data, labels in train_loader:
                data = data.to(device)
                labels = labels.to(device)
                out = model(data)
                loss = criterion(out, labels)
                #print(loss)
                loss.backward()
                #print(model.codebook.grad)
                optimizer.step()
                optimizer.zero_grad()

        return model.codebook.to(device), model.sigma.to(device)


    def forward(self,x):

        x2 = self.activations(self.conv1(x.to(device)).to(device))

        if self.start_bof_training:
            histogram1 = torch.flatten(x2, start_dim = 2, end_dim=3).to(device) #receives output of conv layer OR original input
            histogram1 = histogram1.transpose(1,2)
            histogram1 = histogram1.unsqueeze(1)
            histogram1 = torch.exp(-(histogram1.to(device) - self.codebook1.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma[1].unsqueeze(0).unsqueeze(2).to(device))
            histogram1 = F.normalize(histogram1, 1, dim = 1).to(device)

            histogram1 = torch.transpose(histogram1, 1, 2)

            #histogram1 = torch.mean(histogram1, dim = 2)#if used returns hist for img
            #x2_for_hist1 = x2.clone()#old_calc_through function
            #histogram_1 = calculate_bof_histogram(x2_for_hist1, self.codebook1.to(device),'rbf', self.sigma, number = 1)

        if self.arch1 == True:
            x3 = self.activations(self.conv2(x2.to(device)))
            if self.start_bof_training:
                histogram2 = torch.flatten(x3, start_dim = 2, end_dim=3).to(device) #receives output of conv layer OR original input
                histogram2 = histogram2.transpose(1,2)
                histogram2 = histogram2.unsqueeze(1)
                histogram2 = torch.exp(-(histogram2.to(device) - self.codebook2.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma[2].unsqueeze(0).unsqueeze(2).to(device))
                histogram2 = F.normalize(histogram2, 1, dim = 1).to(device)

                histogram2 = torch.transpose(histogram2, 1, 2)

                #histogram2 = torch.mean(histogram2, dim = 2)               
                #x3_for_hist2 = x3.clone()
                #histogram_2 = calculate_bof_histogram(x3_for_hist2, self.codebook2.to(device),'rbf', self.sigma, number = 2)
            x3 = self.pool1(x3)
        else:
            x3 = self.activations(self.conv2(x2.to(device)))
            if self.start_bof_training:
                histogram2 = torch.flatten(x3, start_dim = 2, end_dim=3).to(device) #receives output of conv layer OR original input
                histogram2 = histogram2.transpose(1,2)
                histogram2 = histogram2.unsqueeze(1)
                histogram2 = torch.exp(-(histogram2.to(device) - self.codebook2.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma[2].unsqueeze(0).unsqueeze(2).to(device))
                histogram2 = F.normalize(histogram2, 1, dim = 1).to(device)

                histogram2 = torch.transpose(histogram2, 1, 2)

                #histogram2 = torch.mean(histogram2, dim = 2)
                #x3_for_hist2 = x3.clone()
                #histogram_2 = calculate_bof_histogram(x3_for_hist2, self.codebook2.to(device),'rbf', self.sigma, number = 2)
            x3 = self.pool2(x3)
        
        x4 = self.activations(self.conv3(x3.to(device)))
        if self.start_bof_training:
            histogram3 = torch.flatten(x4, start_dim = 2, end_dim=3).to(device) #receives output of conv layer OR original input
            histogram3 = histogram3.transpose(1,2)
            histogram3 = histogram3.unsqueeze(1)
            histogram3 = torch.exp(-(histogram3.to(device) - self.codebook3.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma[3].unsqueeze(0).unsqueeze(2).to(device))
            histogram3 = F.normalize(histogram3, 1, dim = 1).to(device)

            histogram3 = torch.transpose(histogram3, 1, 2)

            #histogram3 = torch.mean(histogram3, dim = 2)   
            #x4_for_hist3 = x4.clone()
            #histogram_3 = torch.tensor(calculate_bof_histogram(x4_for_hist3, self.codebook3.to(device), 'rbf', self.sigma, number = 3))

        if self.arch1 == False:
            x5 = self.activations(self.conv4(x4.to(device)))
            if self.start_bof_training:
                histogram4 = torch.flatten(x5, start_dim = 2, end_dim=3).to(device) #receives output of conv layer OR original input
                histogram4 = histogram4.transpose(1,2)
                histogram4 = histogram4.unsqueeze(1)
                histogram4 = torch.exp(-(histogram4.to(device) - self.codebook4.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma[4].unsqueeze(0).unsqueeze(2).to(device))
                histogram4 = F.normalize(histogram4, 1, dim = 1).to(device)

                histogram4 = torch.transpose(histogram4, 1, 2)

                #histogram4 = torch.mean(histogram4, dim = 2)
                #x5_for_hist4 = x.clone()
                #histogram_4 = calculate_bof_histogram(x5_for_hist4, self.codebook4.to(device), 'rbf', self.sigma, number = 4)
            x = self.pool2_2(x5)
        else:
            x = self.activations(self.conv4(x4.to(device)))
            if self.start_bof_training:
                histogram4 = torch.flatten(x, start_dim = 2, end_dim=3).to(device) #receives output of conv layer OR original input
                histogram4 = histogram4.transpose(1,2)
                histogram4 = histogram4.unsqueeze(1)
                histogram4 = torch.exp(-(histogram4.to(device) - self.codebook4.unsqueeze(0).unsqueeze(2).to(device)).abs().pow(2).sum(3) * self.sigma[4].unsqueeze(0).unsqueeze(2).to(device))
                histogram4 = F.normalize(histogram4, 1, dim = 1).to(device)

                histogram4 = torch.transpose(histogram4, 1, 2)

                #histogram4 = torch.mean(histogram4, dim = 2)
                #x5_for_hist4 = x.clone()
                #histogram_4 = calculate_bof_histogram(x5_for_hist4, self.codebook4.to(device), 'rbf', self.sigma, number = 4)
        

        x = self.conv5(x.to(device))
        x= torch.flatten(x.to(device).to(device), start_dim=1, end_dim = 3)
        if not self.start_bof_training:
            histogram1 = torch.tensor(0)
            histogram2 = torch.tensor(0)
            histogram3 = torch.tensor(0)
            histogram4 = torch.tensor(0)

        return x, histogram1, histogram2, histogram3, histogram4, x5#added x5 instead of x to run baseline knn test