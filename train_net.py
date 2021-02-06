import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import bof_parallel_net
import data_python
import evaluate_model
import visualization_hidden
import utils

import os
import time
import argparse

device = (torch.device('cuda') if torch.cuda.is_available()
else torch.device('cpu'))
print(f"Training on device {device}.")

save_model = True
load_model = False

torch.manual_seed(20)
np.random.seed(20)

argparser = argparse.ArgumentParser(description='Configure hyper-parameters')
argparser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
argparser.add_argument('--optimizer',  choices = ['sgd','adam','rmsprop','adadelta','adagrad'], default='sgd',   help='Optimizer')
argparser.add_argument('--dataset', type=str,   default='cifar10', help='Dataset. Currently supports cifar10 and mnist')
argparser.add_argument('--batch_size', type=int,   default=250, help='Batch size')
argparser.add_argument('--epochs', type=int,   default=60, help='Epochs')
argparser.add_argument('--arch', type=int,   default=2, help='Choose an architecture')
argparser.add_argument('--eval_freq', type=int,   default=5, help='Calculate accuracy for train and test after __ epochs')
argparser.add_argument('--bof_centers', type=int,   default=20, help='Number of trainable centers to be used by the BOF layer')
argparser.add_argument('--path', type=str,   default="results", help='Path to save the results. It will create a dir')
argparser.add_argument('--exp_number', type=int,   default=0, help='Experiment number if multiple tries are to take place. Add one each time to create a different directory')
argparser.add_argument('--epochs_init', type=int,   default=12, help='Epochs before training bof centers')
argparser.add_argument('--augmentation', type=bool,   default=False, help='Apply data augmentation to cifar')
argparser.add_argument('--histogram_to_transfer', type=int,   default=1, help='Choose a histogram level to transform knowledge. Choose 0 if gradual.')

args = argparser.parse_args()

if args.dataset == 'cifar10':
    train_loader, test_loader, train_original = data_python.cifar10_loader(data_path='data', batch_size=args.batch_size, augment_train= args.augmentation)
if args.dataset == 'mnist':
    train_loader, test_loader, train_original = data_python.mnist_loader(data_path='data', batch_size=args.batch_size)

#This for samples a batch randomly to initialize bof layers
for data,lab in train_original:
  bof_cents = data
  bof_targs = lab
  break

os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(args.exp_number))
os.system("mkdir " + args.path + "/experiment_" + str(args.exp_number) + '/bof_histograms/')


#=====================================#
#Code below is used to train a teacher - can be skipped if we assume teacher has been trained and resides at path given in load
#=====================================#
#Initialize the network. This is required irregardless of whether or not we train
teacher = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),
 clusters = args.bof_centers, arch = args.arch, quant_input = True, end_with_linear = False,
 activation = 'relu', path = args.path, exp_number = args.exp_number)
teacher.to(device)

# Optimizer
#if args.optimizer == 'sgd':
#    optimizer = torch.optim.SGD(student.parameters(),lr= args.lr)
#elif args.optimizer == 'rmsprop':
#    optimizer = torch.optim.RMSprop(student.parameters(),lr= args.lr)
#elif args.optimizer == 'adadelta':
#    optimizer = torch.optim.Adadelta(student.parameters(),lr= args.lr)
#elif args.optimizer == 'adagrad':
#    optimizer = torch.optim.Adagrad(student.parameters(),lr= args.lr)
#elif args.optimizer == 'adam':
#    optimizer = torch.optim.Adam(student.parameters(),lr= args.lr)

#start = time.time()
##Classification loss i.e. cross entropy loss
#criterion = nn.CrossEntropyLoss()
#
##Train_original is used if image augmentation takes place
#utils.train_bof_model(teacher, optimizer, criterion, train_loader, train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
#args.path, args.exp_number, 100, 200)
#
##Saves the model in a model.pt file after the end of args.epochs epochs
#torch.save(teacher.state_dict(), args.path + "/experiment_" + str(args.exp_number) + "/model.pt")
#
#end = time.time()
#print("Teacher training time: ", end - start, "sec")

#==================================#
#If teacher is trained and we are only interested in loading it
#==================================#
model_dict = teacher.state_dict()
pretrained = torch.load('/content/drive/MyDrive/experiments_information_theory/Plots/experiment_325/model.pt')
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
teacher.load_state_dict(model_dict)

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),
 clusters = args.bof_centers, arch = args.arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = args.exp_number)
student.to(device)
student.student_network = True

# Optimizer
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(student.parameters(),lr= args.lr)
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(student.parameters(),lr= args.lr)
elif args.optimizer == 'adadelta':
    optimizer = torch.optim.Adadelta(student.parameters(),lr= args.lr)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(student.parameters(),lr= args.lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(student.parameters(),lr= args.lr)

start = time.time()
#Classification loss i.e. cross entropy loss
criterion = nn.CrossEntropyLoss()

#Train_original is used if image augmentation takes place // had it to return accuracies instead of void to save everything in log
train_acc_list, test_acc_list = utils.train_bof_for_kt(student, teacher, optimizer, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, args.exp_number, 500, 220, args.histogram_to_transfer)

#Saves the model in a model.pt file after the end of args.epochs epochs
torch.save(student.state_dict(), args.path + "/experiment_" + str(args.exp_number) + "/model_after_transfer.pt")

end = time.time()
print("Student training time: ", end - start, "sec")

with open(args.path + '/experiment_' + str(args.exp_number) + '/params.txt', 'w') as f:
    f.write("Epochs: ")
    f.write(str(args.epochs))
    f.write("\n")
    f.write("Notes: ")
    f.write(' ')
    f.write("Dataset: ")
    f.write(args.dataset)
    f.write("\n")
    f.write("arch: ")
    f.write(str(args.arch))
    f.write("\n")
    f.write("net transfer level: ")
    f.write(str(args.histogram_to_transfer))
    f.write("\n")
    f.write("Train acc: ")
    f.write(str(train_acc_list))
    f.write("\n")
    f.write("Test acc: ")
    f.write(str(test_acc_list))
    f.write("\n")
    f.close()