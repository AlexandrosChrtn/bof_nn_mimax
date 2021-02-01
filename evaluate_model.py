import torch
import numpy as np
from sklearn.metrics import accuracy_score

device = (torch.device('cuda') if torch.cuda.is_available()
else torch.device('cpu'))

def evaluate_model_train_test(net, train_loader, test_loader, train_list, test_list):
    train_accuracy_list_monitor = []
    test_accuracy_list_monitor = []
    for instances, labels in test_loader:
      instances.cuda()
      labels.cuda()
      labels = labels.type(torch.LongTensor)
      predict_out = net(instances)
      _, predict_y2 = torch.max(predict_out[0], 1)
      acctest = accuracy_score(labels.cpu().data, predict_y2.cpu().data)
      test_accuracy_list_monitor.append(acctest)
    for instances, labels in train_loader:
      instances.cuda()
      labels.cuda()
      labels = labels.type(torch.LongTensor)
      predict_out = net(instances)
      _, predict_y2 = torch.max(predict_out[0], 1)
      acctrain = accuracy_score(labels.cpu().data, predict_y2.cpu().data)
      train_accuracy_list_monitor.append(acctrain)
    test_mean_acc = np.mean(np.array(test_accuracy_list_monitor))
    train_mean_acc = np.mean(np.array(train_accuracy_list_monitor))
    test_list.append(test_mean_acc)
    train_list.append(train_mean_acc)
    print('train: ', train_mean_acc)
    print('test ', test_mean_acc)
    return train_mean_acc,  test_mean_acc
