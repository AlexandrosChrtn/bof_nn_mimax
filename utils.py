import numpy as np
import torch
import evaluate_model
from visualization_hidden import plot_accuracies, plot_loss, mkdir_and_vis_hist, plot_mi
device = (torch.device('cuda') if torch.cuda.is_available()
else torch.device('cpu'))
from mi_estimation import mi_between_quantized


def train_bof_model(net, optimizer, criterion, train_loader, train_loader_original, test_loader, epoch_to_init, epochs, eval_freq, path, exp_number, k_means_iter, codebook_iter):
    """
    Trains a classification model
    :param net: model to train - object that returns output along with histograms - convbofvgg object
    :param optimizer: optimizer to minimize loss
    :param criterion: loss
    :param train_loader: pytorch object that feeds instance-label pairs
    :param test_loader: pytorch object that feeds instance-label pairs from test
    :param epochs: epochs
    :param eval_freq: evaluatation frequency
    :return: returns void -- trains the input model based on data from trainloader and plots accuracies in train and test
    and the loss during training
    """
    train_accuracy = []
    test_accuracy = []
    ce_loss = []
    for epoch in range(epochs):
        net.train()
        train_loss, correct, total = 0.0, 0.0, 0.0
        if epoch == (epoch_to_init - 1):
            net.prepare_centers(k_means_iter,codebook_iter)

        for (instances, labels) in train_loader:
            instances, labels = instances.to(device), labels.to(device)

            #may need the following line
            #labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            out, hist1, hist2, hist3, hist4 = net(instances)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # Adds calculated loss to total loss and the maximum output for the prediction
            train_loss += loss.data.item() / labels.size(0)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()
            
            #Code below is used to plot the extracted histograms every 15 epochs to ensure everything works
            #if epoch % 15 == 0 and epoch > 1:
            #    hist0 = torch.mean(hist0, dim = 1)
            #    hist1 = torch.mean(hist1, dim = 1)
            #    hist2 = torch.mean(hist2, dim = 1)
            #    hist3 = torch.mean(hist3, dim = 1)
            #    hist4 = torch.mean(hist4, dim = 1)
            #    mkdir_and_vis_hist(hist0.cpu(), hist1.cpu(), hist2.cpu(), hist3.cpu(), hist4.cpu(),labels.cpu(), path, exp_number,epoch)
        
        ce_loss.append(train_loss / total)
        #code below is repsonsible for evaluating every freq eval epochs
        if epoch > 1 and (epoch % eval_freq == 0 or epoch == epochs - 1):
            evaluate_model.evaluate_model_train_test(net, train_loader_original, test_loader, train_accuracy, test_accuracy)
       

        print("\nLoss, acc = ", train_loss, correct / total)
    plot_accuracies(train_accuracy = train_accuracy, test_accuracy = test_accuracy, path = path, experiment_number = exp_number, epochs = epochs)
    plot_loss(loss = ce_loss, experiment_number = exp_number, path = path, epochs = epochs)


def train_bof_for_kt(student, teacher, optimizer, criterion, train_loader, train_loader_original, test_loader, epoch_to_init, epochs, eval_freq, path, exp_number, k_means_iter, codebook_iter, histogram_to_transfer):
    """
    Trains a classification model
    :param student: model to train using knowledge from teacher
    :param teacher: model used as teacher to train student -- so far assuming it had its codebook trained
    :param optimizer: optimizer to minimize loss
    :param criterion: loss
    :param train_loader: pytorch object that feeds instance-label pairs
    :param test_loader: pytorch object that feeds instance-label pairs from test
    :param epochs: epochs
    :param eval_freq: evaluatation frequency
    :return: returns void -- trains the input model based on data from trainloader and plots accuracies in train and test
    and the loss during training
    """
    train_accuracy = []
    test_accuracy = []
    ce_loss = []
    mi_loss = []
    for param in teacher.parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        student.train()
        train_loss, correct, total, calculated_mi = 0.0, 0.0, 0.0, 0.0
        if epoch == (epoch_to_init - 1):
            student.prepare_centers(k_means_iter,codebook_iter)
            teacher.prepare_centers(k_means_iter,codebook_iter) #perhaps do that with a properly trained teacher
            student.start_bof_training = True
            teacher.start_bof_training = True

        for (instances, labels) in train_loader:
            instances, labels = instances.to(device), labels.to(device)

            optimizer.zero_grad()
            out, hist1, hist2, hist3, hist4 = student(instances)
            out_teacher, hist1_teacher, hist2_teacher, hist3_teacher, hist4_teacher = teacher(instances)

            #Ugly code but whatever works for now
            if histogram_to_transfer == 0:
                if epoch < epoch_to_init + 20:
                    vessel, vessel_teacher = hist1, hist1_teacher
                if epoch >= epoch_to_init + 20 and epoch < epoch_to_init + 40:
                    vessel, vessel_teacher = hist2, hist2_teacher
                if epoch >= epoch_to_init + 40:
                    vessel, vessel_teacher = hist3, hist3_teacher
            if histogram_to_transfer == 1:
                vessel, vessel_teacher = hist1, hist1_teacher
            if histogram_to_transfer == 2:
                vessel, vessel_teacher = hist2, hist2_teacher
            if histogram_to_transfer == 3:
                vessel, vessel_teacher = hist3, hist3_teacher

            if epoch < epoch_to_init - 1:
                loss = criterion(out, labels)
                loss.backward()
            else:
                loss1 = criterion(out, labels)
                loss2 = mi_between_quantized(vessel, vessel_teacher)
                loss = loss1 - 0.5 * loss2
                loss.backward(retain_graph=True)
            
            optimizer.step()

            # Adds calculated loss to total loss and the maximum output for the prediction
            if epoch < epoch_to_init - 1:
              train_loss += loss.data.item() / labels.size(0)
            else:
              train_loss += loss1.data.item() / labels.size(0)
              calculated_mi += loss2.data.item() / labels.size(0)

            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

            #Code below is used to inspect odd mi values 
            #if epoch == 18 and epoch > 1:
            #    hist1 = torch.mean(hist1, dim = 1)
            #    hist2 = torch.mean(hist2, dim = 1)
            #    hist3 = torch.mean(hist3, dim = 1)
            #    hist4 = torch.mean(hist4, dim = 1)
            #    mkdir_and_vis_hist(hist1.cpu(), hist2.cpu(), hist3.cpu(), hist4.cpu(),labels.cpu(), path, exp_number,epoch)

        ce_loss.append(train_loss)
        
        if epoch >= epoch_to_init - 1:
            mi_loss.append(calculated_mi)
            #mi_loss.append(loss2.data.item())
        #code below is repsonsible for evaluating every freq eval epochs
        if epoch > 1 and (epoch % eval_freq == 0 or epoch == epochs - 1):
            evaluate_model.evaluate_model_train_test(student, train_loader_original, test_loader, train_accuracy, test_accuracy)
       

        print("\nLoss, acc = ", train_loss, correct / total, 'for epoch ', epoch + 1)
    plot_accuracies(train_accuracy = train_accuracy, test_accuracy = test_accuracy, path = path, experiment_number = exp_number, epochs = epochs)
    plot_loss(loss = ce_loss, experiment_number = exp_number, path = path, epochs = epochs)
    plot_mi(mi = mi_loss, experiment_number = exp_number, path = path, epochs = epochs)
    return train_accuracy, test_accuracy
