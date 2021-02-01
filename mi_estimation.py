import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse, unique_counts

#====================================================================================================================

def qmi(hidden, discrete, sigma=1.52, kernel = 'cauchy', M=4, eps=1e-8, use_square_clamp=False):
    """
    Implements the QSMI loss
    :param documents: the documents representation
    :param targets: the documents labels
    :param sigma: scaling factor for the Gaussian kernel (if used)
    :param eps: a small number used to ensure the stability of the cosine similarity
    :param M: number of information needs (assuming that each one is equiprobable)
    :param use_cosine: Set to true to use QSMI, otherwise QMI is used
    :param use_square_clamp: Set to true to used the square clamping method
    :return: the QMI/QSMI loss
    """
    hidden = torch.Tensor(hidden).to(device)
    discrete = torch.Tensor(discrete).to(device)

    discretenp = discrete.cpu().numpy()
    discrete_dist, discrete, _ = (get_unique_probs(discretenp))

    discrete = torch.Tensor(discrete).to(device)

    if kernel == 'cosine':
        hidden = hidden / (torch.sqrt(torch.sum(hidden ** 2, dim=1, keepdim=True)) + eps)
        Y = torch.mm(hidden, hidden.t()).to(device)
        Y = 0.75 * (Y + 1)
    elif kernel == 'euclidean':
        Y = squared_pairwise_distances(hidden).to(device)
        Y = torch.sqrt(Y)
        Y = 1 / (1 + Y * sigma)
    elif kernel == 'combined':
        Y = squared_pairwise_distances(hidden).to(device)
        Y = 0.5 * (1 / (1 + Y * 4)) + 0.5 * torch.exp(-Y / (2 * sigma ** 2))
    elif kernel == 'cauchy':
        Y = squared_pairwise_distances(hidden).to(device)
        Y = torch.sqrt(Y)
        Y = 1 / (1 + Y / sigma ** 2)
    elif kernel == 'rbf':
        Y = squared_pairwise_distances(hidden).to(device)
        Y = torch.sqrt(Y)
        Y = torch.exp(-Y / (2 * sigma ** 2))

    D = (discrete.view(discrete.shape[0], 1) == discrete.view(1, discrete.shape[0]))

    Q_in = torch.sum(D * Y) / hidden.size(0) **2
    #Q_ALL
    Q_all_coef = (torch.sum(torch.Tensor(discrete_dist ** 2))) / (hidden.size(0) ** 2)
    Q_all = Q_all_coef * torch.sum(Y)

    #Q_BTW
    discrete = discrete.to(device)
    D = D.to(device)
    D = (discrete.view(discrete.shape[0], 1) == torch.zeros(size = (discrete.view(discrete.shape[0], 1).size())).to(device))
    rr = torch.zeros(size = (discrete.view(discrete.shape[0], 1).size()))
    Q_btw = []
    
    for i in range(len(torch.unique(discrete))):
        D = (discrete.view(discrete.shape[0], 1) == rr.to(device)).to(device)
        rr = rr.to(device) + torch.ones(size = (discrete.view(discrete.shape[0], 1).size())).to(device)
        D = D.repeat(1,D.size(0))
        D = D.view(-1,D.size(0), D.size(0)) 
        Q_btw.append((torch.sum(D*Y) * discrete_dist[i]))
    Q_btw = torch.Tensor(Q_btw)

    Q_btw = torch.sum(Q_btw) / (hidden.size(0) ** 2)

    return (Q_in + Q_all).cpu()# - 2 * Q_btw).cpu()


def squared_pairwise_distances(a, b=None):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return: distance matrix 
    """
    if b is None:
        b = a

    aa = torch.sum(a ** 2, dim=1).to(device)
    bb = torch.sum(b ** 2, dim=1).to(device)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1)).to(device)

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf).to(device)
 
    return dists.to(device)

def bucketize(tensor, bins):
    result = torch.zeros_like(tensor, dtype=torch.int32)
    boundaries = torch.arange(0,1, 1/int(bins))
    for boundary in boundaries:
        result += (tensor > boundary).int()
    print(result.float())
    return result.float()

def compute_distances(x):
    '''
    Computes the distance matrix for the KDE Entropy estimation:
    - x (Tensor) : array of functions to compute the distances matrix from
    '''

    x_norm = (x**2).sum(1).view(-1,1)
    x_t = torch.transpose(x,0,1)
    x_t_norm = x_norm.view(1,-1)
    dist = x_norm + x_t_norm - 2.0*torch.mm(x,x_t)
    dist = torch.clamp(dist,0,np.inf)

    return dist

def KDE_IXT_estimation(logvar_t, mean_t):
    '''
    Computes the MI estimation of X and T. Parameters:
    - logvar_t (float) : log(var) of the bottleneck variable 
    - mean_t (Tensor) : deterministic transformation of the input 
    '''
    n_batch, _ = mean_t.shape
    var = torch.exp(logvar_t) + 1e-10 # to avoid 0's in the log

    # calculation of the constant
    normalization_constant = math.log(n_batch)

    # calculation of the elements contribution
    dist = compute_distances(mean_t)
    distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var,dim=1))

    # mutual information calculation (natts)
    I_XT = normalization_constant + distance_contribution

    return I_XT

def csqmi(a, b, kernel = 'rbf', sigma = 1, eps=1e-8):
    # l spans the variables so l represents either a or b -- a and b are continuous
    # i, j span the batch 
    # Y is n x n kernel matrix for a
    # X is n x n kernel matrix for b
    # XY is their hadamard product
    if kernel == 'rbf':
        Y = squared_pairwise_distances(a).to(device)
        Y = torch.sqrt(Y)
        Y = torch.exp(-Y / (2 * sigma ** 2))
        X = squared_pairwise_distances(b).to(device)
        X = torch.sqrt(X)
        X = torch.exp(-X / (2 * sigma ** 2))
        XY = X * Y
    if kernel == 'cosine':
        a = a / (torch.sqrt(torch.sum(a ** 2, dim=1, keepdim=True)) + eps)
        Y = torch.mm(a, a.t()).to(device)
        Y = 0.5 * (Y + 1)
        b = b / (torch.sqrt(torch.sum(b ** 2, dim=1, keepdim=True)) + eps)
        X = torch.mm(b, b.t()).to(device)
        X = 0.5 * (X + 1)
        XY = X * Y

    #V_y is the joint information potential in the joint space
    V_y = (1 / (a.size(0) ** 2)) * torch.sum(XY)

    #V_l_yj_yl is the marginal information potential, with l defining its corresponding marginal field
    V_1_yj_yl = torch.mean(Y, dim = 1)
    V_2_yj_yl = torch.mean(X, dim = 1)

    #V_l_yj is the lth marginal information potential, as it averages every MIP for the variable l
    V_1_y1 = torch.mean(V_1_yj_yl)
    V_2_y2 = torch.mean(V_2_yj_yl)

    #Vnc is the un-normalized cross-information potential and it measures interactions between partial marginal information potentials
    Vnc = torch.mean(V_1_yj_yl*V_2_yj_yl)
    return torch.log((V_y * V_1_y1 * V_2_y2) / (Vnc ** 2))

def mi_between_quantized(membership1, membership2):
    py_given_x = torch.mean(membership1, dim = 1)
    pz_given_x = torch.mean(membership2, dim = 1)
    Bsize = membership1.size(0) #assuming 0 is batch size and 2 is the num of clusters which is equal to both
    Nk = membership1.size(2)
    membership1 = membership1.view(Bsize, -1, Nk, 1)
    membership2 = membership2.view(Bsize, -1, 1, Nk)
    joint_histogram = membership1 * membership2
    joint_histogram = torch.mean(joint_histogram, dim=1)
    
    # Estimate the MI
    marginal_product = (py_given_x.view(Bsize,-1, 1) * pz_given_x.view(Bsize,1, -1))
    mi = joint_histogram * torch.log(1e-7 + joint_histogram / (marginal_product + 1e-7))
    mi = torch.sum(mi, dim = (1,2))

    #return mean for batch -- log with base 'e'
    return torch.mean(mi)
