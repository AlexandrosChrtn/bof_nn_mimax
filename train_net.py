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

# study these later
#torch.backends.cudnn.deterministic = False
#torch.backends.cudnn.benchmark = False 


torch.manual_seed(20)
np.random.seed(20)

argparser = argparse.ArgumentParser(description='Configure hyper-parameters')
argparser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
argparser.add_argument('--optimizer',  choices = ['sgd','adam','rmsprop','adadelta','adagrad'], default='adam',   help='Optimizer')
argparser.add_argument('--dataset', type=str,   default='cifar10', help='Dataset. Currently supports cifar10 and mnist')
argparser.add_argument('--batch_size', type=int,   default=64, help='Batch size')
argparser.add_argument('--epochs', type=int,   default=60, help='Epochs')
argparser.add_argument('--k_means_iter', type=int,   default=500, help='K-means iterations')
argparser.add_argument('--codebook_train_epochs', type=int,   default=50, help='Iterations over initializing batch sample for codebook training')
argparser.add_argument('--student_arch', type=int,   default=2, help='Choose an architecture for student')
argparser.add_argument('--teacher_arch', type=int,   default=3, help='Choose an architecture for teacher')
argparser.add_argument('--eval_freq', type=int,   default=5, help='Calculate accuracy for train and test after __ epochs')
argparser.add_argument('--bof_centers', type=int,   default=20, help='Number of trainable centers to be used by the BOF layer')
argparser.add_argument('--path', type=str,   default="results", help='Path to save the results. It will create a dir')
argparser.add_argument('--exp_number', type=int,   default=0, help='Experiment number if multiple tries are to take place. Add one each time to create a different directory')
argparser.add_argument('--epochs_init', type=int,   default=12, help='Epochs before training bof centers')
argparser.add_argument('--augmentation', type=bool,   default=False, help='Apply data augmentation to cifar')
argparser.add_argument('--histogram_to_transfer', type=int,   default=1, help='Choose a histogram level to transform knowledge. Choose 0 if gradual.')
argparser.add_argument('--load_model_path', type=str,   default="model.pt", help='Path leading to pretrained model.')

args = argparser.parse_args()

if args.dataset == 'cifar10':
    train_loader, test_loader, train_original, train_subset_loader, bof_center_loader = data_python.cifar10_loader(data_path='data', batch_size=args.batch_size, augment_train= args.augmentation)
if args.dataset == 'mnist':
    train_loader, test_loader, train_original = data_python.mnist_loader(data_path='data', batch_size=args.batch_size)

#This for samples a batch randomly to initialize bof layers -- 24-2-2021 uses 500 instances
for data,lab in bof_center_loader:
  bof_cents = data
  bof_targs = lab
  break

os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(args.exp_number))
os.system("mkdir " + args.path + "/experiment_" + str(args.exp_number) + '/bof_histograms/')

#09-03 changed use_hists to 4 for all histogram_to_transfer to measure MI between every hist pair
use_hists = 4

#=====================================#
#Code below is used to train a teacher - can be skipped if we assume teacher has been trained and resides at path given in load
#=====================================#
#Initialize the network. This is required irregardless of whether or not we train -- as per 24-2 we use a fixed arch 2 teacher
teacher = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device), center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.teacher_arch, quant_input = True, end_with_linear = False,
 activation = 'relu', path = args.path, exp_number = args.exp_number, use_hists=use_hists)
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
pretrained = torch.load(args.load_model_path, map_location= device)
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
teacher.load_state_dict(model_dict)

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),  center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.student_arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = args.exp_number, use_hists=use_hists)
student.to(device)
student.student_network = True

#NOTE: The use of a different optimizer with potentially different lr for the centers may be used later. Anycase the argument optimizer_for_centers in the followin train_bof_for_kt func
# will remain there for now
non_bof_params = [student.conv1.weight, student.conv1.bias, student.conv2.weight, student.conv2.bias, student.conv3.weight, student.conv3.bias, 
student.conv4.weight, student.conv4.bias, student.conv5.weight, student.conv5.bias]
optimizer_for_centers = torch.optim.Adam([student.codebook1, student.codebook2, student.codebook3, student.codebook4, student.sigma], lr=0.001)

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
train_acc_list, test_acc_list, accuracy_saver = utils.train_bof_for_kt(student, teacher, optimizer, optimizer_for_centers, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, args.exp_number, args.k_means_iter, args.codebook_train_epochs, args.histogram_to_transfer, coef1 = 0.1, coef2 = 0.1, coef3 = 0.1, coef4 = 0.1,train_hist_for_teacher = True)

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
    f.write(str(args.student_arch))
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
    f.write("Accuracy saver: ")
    f.write(str(accuracy_saver))
    f.write("\n")
    f.write("epochs_init: ")
    f.write(str(args.epochs_init))
    f.write("\n")
    f.write("teacher path: ")
    f.write((args.load_model_path))
    f.write("\n")
    f.write("Teacher_arch: ")
    f.write(str(args.teacher_arch))
    f.write("\n")
    f.close()
#=========================================

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
exp2 = args.exp_number + 1
os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(exp2))
os.system("mkdir " + args.path + "/experiment_" + str(exp2) + '/bof_histograms/')

student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),  center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.student_arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = exp2, use_hists=use_hists)
student.to(device)
student.student_network = True

#NOTE: The use of a different optimizer with potentially different lr for the centers may be used later. Anycase the argument optimizer_for_centers in the followin train_bof_for_kt func
# will remain there for now
non_bof_params = [student.conv1.weight, student.conv1.bias, student.conv2.weight, student.conv2.bias, student.conv3.weight, student.conv3.bias, 
student.conv4.weight, student.conv4.bias, student.conv5.weight, student.conv5.bias]
optimizer_for_centers = torch.optim.Adam([student.codebook1, student.codebook2, student.codebook3, student.codebook4, student.sigma], lr=0.001)

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
train_acc_list, test_acc_list, accuracy_saver = utils.train_bof_for_kt(student, teacher, optimizer, optimizer_for_centers, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, exp2, args.k_means_iter, args.codebook_train_epochs, args.histogram_to_transfer, coef1 = 0.25, coef2 = 0.25, coef3 = 0.25, coef4 = 0.25)

#Saves the model in a model.pt file after the end of args.epochs epochs
torch.save(student.state_dict(), args.path + "/experiment_" + str(exp2) + "/model_after_transfer.pt")

end = time.time()
print("Student training time: ", end - start, "sec")

with open(args.path + '/experiment_' + str(exp2) + '/params.txt', 'w') as f:
    f.write("Epochs: ")
    f.write(str(args.epochs))
    f.write("\n")
    f.write("Notes: ")
    f.write(' ')
    f.write("Dataset: ")
    f.write(args.dataset)
    f.write("\n")
    f.write("arch: ")
    f.write(str(args.student_arch))
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
    f.write("Accuracy saver: ")
    f.write(str(accuracy_saver))
    f.write("\n")
    f.write("epochs_init: ")
    f.write(str(args.epochs_init))
    f.write("\n")
    f.write("teacher path: ")
    f.write((args.load_model_path))
    f.write("\n")
    f.write("Teacher_arch: ")
    f.write(str(args.teacher_arch))
    f.write("\n")
    f.close()
#=========================================

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
exp3 = args.exp_number + 3
os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(exp3))
os.system("mkdir " + args.path + "/experiment_" + str(exp3) + '/bof_histograms/')


student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),  center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.student_arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = exp3, use_hists=use_hists)
student.to(device)
student.student_network = True

#NOTE: The use of a different optimizer with potentially different lr for the centers may be used later. Anycase the argument optimizer_for_centers in the followin train_bof_for_kt func
# will remain there for now
non_bof_params = [student.conv1.weight, student.conv1.bias, student.conv2.weight, student.conv2.bias, student.conv3.weight, student.conv3.bias, 
student.conv4.weight, student.conv4.bias, student.conv5.weight, student.conv5.bias]
optimizer_for_centers = torch.optim.Adam([student.codebook1, student.codebook2, student.codebook3, student.codebook4, student.sigma], lr=0.001)

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
train_acc_list, test_acc_list, accuracy_saver = utils.train_bof_for_kt(student, teacher, optimizer, optimizer_for_centers, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, exp3, args.k_means_iter, args.codebook_train_epochs, args.histogram_to_transfer, coef1 = 0.01, coef2 = 0.01, coef3 = 0.01, coef4 = 0.01)

#Saves the model in a model.pt file after the end of args.epochs epochs
torch.save(student.state_dict(), args.path + "/experiment_" + str(exp3) + "/model_after_transfer.pt")

end = time.time()
print("Student training time: ", end - start, "sec")

with open(args.path + '/experiment_' + str(exp3) + '/params.txt', 'w') as f:
    f.write("Epochs: ")
    f.write(str(args.epochs))
    f.write("\n")
    f.write("Notes: ")
    f.write(' ')
    f.write("Dataset: ")
    f.write(args.dataset)
    f.write("\n")
    f.write("arch: ")
    f.write(str(args.student_arch))
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
    f.write("Accuracy saver: ")
    f.write(str(accuracy_saver))
    f.write("\n")
    f.write("epochs_init: ")
    f.write(str(args.epochs_init))
    f.write("\n")
    f.write("teacher path: ")
    f.write((args.load_model_path))
    f.write("\n")
    f.write("Teacher_arch: ")
    f.write(str(args.teacher_arch))
    f.write("\n")
    f.close()
#=========================================

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
exp4 = args.exp_number + 4
os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(exp4))
os.system("mkdir " + args.path + "/experiment_" + str(exp4) + '/bof_histograms/')

student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),  center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.student_arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = exp4, use_hists=use_hists)
student.to(device)
student.student_network = True

#NOTE: The use of a different optimizer with potentially different lr for the centers may be used later. Anycase the argument optimizer_for_centers in the followin train_bof_for_kt func
# will remain there for now
non_bof_params = [student.conv1.weight, student.conv1.bias, student.conv2.weight, student.conv2.bias, student.conv3.weight, student.conv3.bias, 
student.conv4.weight, student.conv4.bias, student.conv5.weight, student.conv5.bias]
optimizer_for_centers = torch.optim.Adam([student.codebook1, student.codebook2, student.codebook3, student.codebook4, student.sigma], lr=0.001)

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
train_acc_list, test_acc_list, accuracy_saver = utils.train_bof_for_kt(student, teacher, optimizer, optimizer_for_centers, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, exp4, args.k_means_iter, args.codebook_train_epochs, args.histogram_to_transfer, coef1 = 0.001, coef2 = 0.001, coef3 = 0.001, coef4 = 0.001)

#Saves the model in a model.pt file after the end of args.epochs epochs
torch.save(student.state_dict(), args.path + "/experiment_" + str(exp4) + "/model_after_transfer.pt")

end = time.time()
print("Student training time: ", end - start, "sec")

with open(args.path + '/experiment_' + str(exp4) + '/params.txt', 'w') as f:
    f.write("Epochs: ")
    f.write(str(args.epochs))
    f.write("\n")
    f.write("Notes: ")
    f.write(' ')
    f.write("Dataset: ")
    f.write(args.dataset)
    f.write("\n")
    f.write("arch: ")
    f.write(str(args.student_arch))
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
    f.write("Accuracy saver: ")
    f.write(str(accuracy_saver))
    f.write("\n")
    f.write("epochs_init: ")
    f.write(str(args.epochs_init))
    f.write("\n")
    f.write("teacher path: ")
    f.write((args.load_model_path))
    f.write("\n")
    f.write("Teacher_arch: ")
    f.write(str(args.teacher_arch))
    f.write("\n")
    f.close()
#=========================================

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
exp5 = args.exp_number + 5
os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(exp5))
os.system("mkdir " + args.path + "/experiment_" + str(exp5) + '/bof_histograms/')

student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),  center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.student_arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = exp5, use_hists=use_hists)
student.to(device)
student.student_network = True

#NOTE: The use of a different optimizer with potentially different lr for the centers may be used later. Anycase the argument optimizer_for_centers in the followin train_bof_for_kt func
# will remain there for now
non_bof_params = [student.conv1.weight, student.conv1.bias, student.conv2.weight, student.conv2.bias, student.conv3.weight, student.conv3.bias, 
student.conv4.weight, student.conv4.bias, student.conv5.weight, student.conv5.bias]
optimizer_for_centers = torch.optim.Adam([student.codebook1, student.codebook2, student.codebook3, student.codebook4, student.sigma], lr=0.001)

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
train_acc_list, test_acc_list, accuracy_saver = utils.train_bof_for_kt(student, teacher, optimizer, optimizer_for_centers, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, exp5, args.k_means_iter, args.codebook_train_epochs, args.histogram_to_transfer, coef1 = 0.8, coef2 = 0.8, coef3 = 0.8, coef4 = 0.8)

#Saves the model in a model.pt file after the end of args.epochs epochs
torch.save(student.state_dict(), args.path + "/experiment_" + str(exp5) + "/model_after_transfer.pt")

end = time.time()
print("Student training time: ", end - start, "sec")

with open(args.path + '/experiment_' + str(exp5) + '/params.txt', 'w') as f:
    f.write("Epochs: ")
    f.write(str(args.epochs))
    f.write("\n")
    f.write("Notes: ")
    f.write(' ')
    f.write("Dataset: ")
    f.write(args.dataset)
    f.write("\n")
    f.write("arch: ")
    f.write(str(args.student_arch))
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
    f.write("Accuracy saver: ")
    f.write(str(accuracy_saver))
    f.write("\n")
    f.write("epochs_init: ")
    f.write(str(args.epochs_init))
    f.write("\n")
    f.write("teacher path: ")
    f.write((args.load_model_path))
    f.write("\n")
    f.write("Teacher_arch: ")
    f.write(str(args.teacher_arch))
    f.write("\n")
    f.close()
#=========================================

#==================================#
#Code below is used to train the student using the quantized representation of the teacher in hist 3
#==================================#
exp6 = args.exp_number + 6
os.system("mkdir " + args.path)
os.system("mkdir " + args.path + "/experiment_" + str(exp6))
os.system("mkdir " + args.path + "/experiment_" + str(exp6) + '/bof_histograms/')

student = bof_parallel_net.ConvBOFVGG(center_initial = bof_cents.to(device), center_initial_y = bof_targs.to(device),  center_train = train_subset_loader,
 clusters = args.bof_centers, arch = args.student_arch, quant_input = True, end_with_linear = False,
 activation = 'sin', path = args.path, exp_number = exp6, use_hists=use_hists)
student.to(device)
student.student_network = True

#NOTE: The use of a different optimizer with potentially different lr for the centers may be used later. Anycase the argument optimizer_for_centers in the followin train_bof_for_kt func
# will remain there for now
non_bof_params = [student.conv1.weight, student.conv1.bias, student.conv2.weight, student.conv2.bias, student.conv3.weight, student.conv3.bias, 
student.conv4.weight, student.conv4.bias, student.conv5.weight, student.conv5.bias]
optimizer_for_centers = torch.optim.Adam([student.codebook1, student.codebook2, student.codebook3, student.codebook4, student.sigma], lr=0.001)

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
train_acc_list, test_acc_list, accuracy_saver = utils.train_bof_for_kt(student, teacher, optimizer, optimizer_for_centers, criterion, train_loader,
train_original, test_loader, args.epochs_init, args.epochs, args.eval_freq, 
args.path, exp6, args.k_means_iter, args.codebook_train_epochs, args.histogram_to_transfer, coef1 = 1.25, coef2 = 1.25, coef3 = 1.25, coef4 = 1.25)

#Saves the model in a model.pt file after the end of args.epochs epochs
torch.save(student.state_dict(), args.path + "/experiment_" + str(exp6) + "/model_after_transfer.pt")

end = time.time()
print("Student training time: ", end - start, "sec")

with open(args.path + '/experiment_' + str(exp6) + '/params.txt', 'w') as f:
    f.write("Epochs: ")
    f.write(str(args.epochs))
    f.write("\n")
    f.write("Notes: ")
    f.write(' ')
    f.write("Dataset: ")
    f.write(args.dataset)
    f.write("\n")
    f.write("arch: ")
    f.write(str(args.student_arch))
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
    f.write("Accuracy saver: ")
    f.write(str(accuracy_saver))
    f.write("\n")
    f.write("epochs_init: ")
    f.write(str(args.epochs_init))
    f.write("\n")
    f.write("teacher path: ")
    f.write((args.load_model_path))
    f.write("\n")
    f.write("Teacher_arch: ")
    f.write(str(args.teacher_arch))
    f.write("\n")
    f.close()
#=========================================