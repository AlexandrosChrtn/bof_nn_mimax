import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def plot_bottleneck(hidden, classes, path, experiment_number, epoch, number):
    pca = PCA(n_components=2)
    #sscaler = StandardScaler()
    hidden = np.array(hidden)
    #hidden = sscaler.fit_transform(hidden)
    hidden = pca.fit_transform(hidden)
    colors = ['red', 'blue', 'orange', 'black', 'yellow', 'pink', 'green', 'purple', 'brown', 'cyan']
    plt.scatter(x = np.transpose(hidden)[0], y = np.transpose(hidden)[1], c = classes.numpy(),cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('PCA of hidden')
    plt.savefig(path + '/experiment_' + str(experiment_number) + '/bottleneck_layer' + str(number) + '_at_epoch_' + str(epoch))
    plt.clf()

def vis_bof_histograms(histogram, labels, path, experiment_number, bof_number, epoch):
    idxlist = []
    for i in range (0, 10):
        idxlist.append(labels == i)
    for j in range (0,10):
        height = torch.mean(histogram[idxlist[j]], dim = 0).detach().numpy()#torch.mean returns the mean vector for each class
        bars = (np.arange(0,histogram.shape[1], 1)).astype(str)
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.title('Bof Histogram (' + str(bof_number) + ') epoch '+str(epoch))
        plt.savefig(path + '/experiment_' + str(experiment_number) + '/bof_histograms/epoch_'+str(epoch)+'/bof_layer_'+str(bof_number)+'/class_' + str(j) + '_bof_' + str(bof_number) +'_at_epoch' + str(epoch))
        plt.clf()

def mkdir_and_vis_hist(hist1, hist2, hist3, hist4,labels,path, experiment_number,epoch):
    os.system('mkdir ' + path + '/experiment_' + str(experiment_number) + '/bof_histograms/epoch_'+str(epoch))
    for bof_number in range(5):
            os.system('mkdir ' + path + '/experiment_' + str(experiment_number) + '/bof_histograms/epoch_'+str(epoch)+'/bof_layer_'+str(bof_number))

    vis_bof_histograms(hist1, labels, path, experiment_number,1, epoch)
    vis_bof_histograms(hist2, labels, path, experiment_number,2, epoch)
    vis_bof_histograms(hist3, labels, path, experiment_number,3, epoch)
    vis_bof_histograms(hist4, labels, path, experiment_number,4, epoch)

def plot_accuracies(train_accuracy, test_accuracy, path, experiment_number, epochs):
    plt.title('Accuracy train / test')
    xi = (np.arange(5, epochs + 5, 5))
    plt.plot(xi,test_accuracy, label = 'test accuracy')
    plt.plot(xi,train_accuracy, label = 'train accuracy')
    plt.legend()
    plt.savefig(path + '/experiment_' + str(experiment_number) + '/acc_plot')
    plt.clf()

def plot_accuracy_saver(accuracy_saver, path, experiment_number):
    plt.title('Train accuracy per epoch')
    plt.plot(accuracy_saver, label = 'Train accuracy')
    plt.legend()
    plt.savefig(path + '/experiment_' + str(experiment_number) + '/accuracy_every_epoch_plot')
    plt.clf()



def plot_loss(loss, experiment_number, path, epochs):
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(loss)
    plt.savefig(path + '/experiment_' + str(experiment_number) + '/loss')
    plt.clf()

def plot_mi(mi, experiment_number, path, epochs):
    plt.title("MI between quantized")
    plt.xlabel("epochs")
    plt.ylabel("MI")
    plt.plot(mi)
    plt.savefig(path + '/experiment_' + str(experiment_number) + '/mi_loss')
    plt.clf()
