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
      predict_out, _, _, _, _ = net(instances)  #added x5 instead of x to run baseline knn test
      _, predict_y2 = torch.max(predict_out, 1) #removed [0] after predict_out
      acctest = accuracy_score(labels.cpu().data, predict_y2.cpu().data)
      test_accuracy_list_monitor.append(acctest)
    for instances, labels in train_loader:
      instances.cuda()
      labels.cuda()
      labels = labels.type(torch.LongTensor)
      predict_out, _, _, _, _= net(instances)  #added x5 instead of x to run baseline knn test
      _, predict_y2 = torch.max(predict_out, 1)
      acctrain = accuracy_score(labels.cpu().data, predict_y2.cpu().data)
      train_accuracy_list_monitor.append(acctrain)
    test_mean_acc = np.mean(np.array(test_accuracy_list_monitor))
    train_mean_acc = np.mean(np.array(train_accuracy_list_monitor))
    test_list.append(test_mean_acc)
    train_list.append(train_mean_acc)
    print('train: ', train_mean_acc)
    print('test ', test_mean_acc)
    return train_mean_acc,  test_mean_acc

def neural_test_evaluation(model, testloader):
  test_accuracy = []
  for data, labels in testloader:
    data = data.to(device)
    labels = labels.to(device)
    prediction, _, _, _, _ = model(data)
    _, predict_y2 = torch.max(prediction, 1)
    test_accuracy.append(accuracy_score(labels.cpu().data, predict_y2.cpu().data))
  return np.mean(np.array(test_accuracy))

def knn_baseline_evaluation(model, knn_after_fit, testloader):
  print('Running kNN evaluation')
  test_accuracy = []
  for data, labels in testloader:
    data = data.to(device)
    labels = labels.to(device)
    _, _, _, _, hist4 = model(data) #removed , _
    
    hist4 = torch.mean(hist4, dim = 1)
    hist4 = hist4.detach().cpu().numpy()
    knn_prediction = knn_after_fit.predict(hist4)

    test_accuracy.append(accuracy_score(labels.cpu().data, knn_prediction))
  return np.mean(np.array(test_accuracy))
