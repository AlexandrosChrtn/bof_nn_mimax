import numpy as np
import torch
import evaluate_model
from visualization_hidden import plot_accuracies, plot_loss, plot_accuracy_saver, mkdir_and_vis_hist, plot_mi
device = (torch.device('cuda') if torch.cuda.is_available()
else torch.device('cpu'))
from mi_estimation import mi_between_quantized

from sklearn.neighbors import KNeighborsClassifier

def calculate_mi_for_every_hist(hist1,hist2,hist3,hist4,hist1_teacher,hist2_teacher,hist3_teacher,hist4_teacher):
    loss_hist1 = mi_between_quantized(hist1, hist1_teacher)
    loss_hist2 = mi_between_quantized(hist2, hist2_teacher)
    loss_hist3 = mi_between_quantized(hist3, hist3_teacher)
    loss_hist4 = mi_between_quantized(hist4, hist4_teacher)
    return loss_hist1, loss_hist2, loss_hist3, loss_hist4

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
            out, hist1, hist2, hist3, hist4, x5 = net(instances) #added x5 instead of x to run baseline knn test

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # Adds calculated loss to total loss and the maximum output for the prediction
            train_loss += loss.data.item() / labels.size(0)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()
        
        ce_loss.append(train_loss / total)
        #code below is repsonsible for evaluating every freq eval epochs
        if epoch > 1 and (epoch % eval_freq == 0 or epoch == epochs - 1):
            evaluate_model.evaluate_model_train_test(net, train_loader_original, test_loader, train_accuracy, test_accuracy)
       

        print("\nLoss, acc = ", train_loss, correct / total)
    plot_accuracies(train_accuracy = train_accuracy, test_accuracy = test_accuracy, path = path, experiment_number = exp_number, epochs = epochs, eval_freq = eval_freq)
    plot_loss(loss = ce_loss, experiment_number = exp_number, path = path, epochs = epochs)


def train_bof_for_kt(student, teacher, optimizer, bof_params_optimizer, criterion, train_loader, train_loader_original, test_loader, epoch_to_init, epochs,
 eval_freq, path, exp_number, k_means_iter, codebook_iter, histogram_to_transfer, coef1, coef2, coef3, coef4, check_baseline_knn_argument = False):
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
    :return: three lists with accuracies
    """

    train_accuracy = []
    test_accuracy = []
    ce_loss = []
    mi_loss = []
    mi_loss2 = []
    mi_hist1 = []
    mi_hist2 = []
    mi_hist3 = []
    mi_hist4 = []
    accuracy_saver = []
    if check_baseline_knn_argument:
        knn_base = KNeighborsClassifier(n_neighbors = 3)
        accuracy_for_knn = []
    for param in teacher.parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        student.train()
        train_loss, correct, total, calculated_mi, calculated_mi2 = .0, .0, .0, .0, .0
        calculated_mi_for_histogram_1, calculated_mi_for_histogram_2, calculated_mi_for_histogram_3, calculated_mi_for_histogram_4 = .0,.0,.0,.0
        if epoch == (epoch_to_init - 1):
            student.prepare_centers(k_means_iter,codebook_iter)
            teacher.prepare_centers(k_means_iter,codebook_iter)
            student.start_bof_training = True
            teacher.start_bof_training = True
            optimizer = torch.optim.Adam(student.parameters(), lr = 0.0001)

        for (instances, labels) in train_loader:
            instances, labels = instances.to(device), labels.to(device)

            optimizer.zero_grad()
            #bof_params_optimizer.zero_grad()#Currently unavailable

            out, hist1, hist2, hist3, hist4 = student(instances)
            _, hist1_teacher, hist2_teacher, hist3_teacher, hist4_teacher = teacher(instances)

            #Ugly code but whatever works for now
            total_epochs_of_mi_max = 110
            if histogram_to_transfer == 0:
                if epoch < epoch_to_init + 25:
                    vessel, vessel_teacher = hist1, hist1_teacher
                    coef = coef1
                if epoch >= epoch_to_init + 25 and epoch < epoch_to_init + 50:
                    vessel, vessel_teacher = hist2, hist2_teacher
                    coef = coef2
                if epoch >= epoch_to_init + 50 and epoch < epoch_to_init + 75:
                    vessel, vessel_teacher = hist3, hist3_teacher
                    coef = coef3
                if epoch >= epoch_to_init + 75:
                    vessel, vessel_teacher = hist4, hist4_teacher
                    coef = coef4
            if histogram_to_transfer == 5:
                vessel, vessel_teacher = hist1, hist1_teacher
                coef1 = 0.2
                vessel2, vessel_teacher2 = hist2, hist2_teacher
                coef2 = 0.4
            if histogram_to_transfer == 1:
                vessel, vessel_teacher = hist1, hist1_teacher
                coef = 0.05
            if histogram_to_transfer == 2:
                vessel, vessel_teacher = hist2, hist2_teacher
                coef = 0.1
            if histogram_to_transfer == 3:
                vessel, vessel_teacher = hist3, hist3_teacher
                coef = 0.4
            if histogram_to_transfer == 4:
                vessel, vessel_teacher = hist4, hist4_teacher
                coef = 0.6

            if epoch < epoch_to_init - 1 or epoch >= total_epochs_of_mi_max:
                loss = criterion(out, labels)
                loss_hist1, loss_hist2, loss_hist3, loss_hist4 = calculate_mi_for_every_hist(hist1,hist2,hist3,hist4,hist1_teacher,hist2_teacher,hist3_teacher,hist4_teacher)
                loss.backward()
                student.start_bof_training = False
                teacher.start_bof_training = False
            elif epoch >= epoch_to_init - 1 and epoch < total_epochs_of_mi_max and not histogram_to_transfer == 5:
                loss1 = criterion(out, labels)
                loss2 = mi_between_quantized(vessel, vessel_teacher)
                loss = loss1 - coef * loss2
                loss.backward(retain_graph=True)
                loss_hist1, loss_hist2, loss_hist3, loss_hist4 = calculate_mi_for_every_hist(hist1,hist2,hist3,hist4,hist1_teacher,hist2_teacher,hist3_teacher,hist4_teacher)
            else:
                loss1 = criterion(out, labels)
                loss2 = mi_between_quantized(vessel, vessel_teacher)
                loss3 = mi_between_quantized(vessel2, vessel_teacher2)
                loss = loss1 - coef1 * loss2- coef2 * loss3
                loss.backward(retain_graph=True)
                loss_hist1, loss_hist2, loss_hist3, loss_hist4 = calculate_mi_for_every_hist(hist1,hist2,hist3,hist4,hist1_teacher,hist2_teacher,hist3_teacher,hist4_teacher)
            
            optimizer.step()
            #bof_params_optimizer.step()#Currently unavailable

            # Adds calculated loss to total loss and the maximum output for the prediction
            if epoch < epoch_to_init - 1:
              train_loss += loss.data.item() / labels.size(0)
            else:
                train_loss += loss1.data.item() / labels.size(0)
                calculated_mi += loss2.data.item() / labels.size(0)
                if histogram_to_transfer == 5:
                    calculated_mi2 += loss3.data.item() / labels.size(0)
            calculated_mi_for_histogram_1 += loss_hist1.data.item() / labels.size(0)
            calculated_mi_for_histogram_2 += loss_hist2.data.item() / labels.size(0)
            calculated_mi_for_histogram_3 += loss_hist3.data.item() / labels.size(0)
            calculated_mi_for_histogram_4 += loss_hist4.data.item() / labels.size(0)
                
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

            if check_baseline_knn_argument and epoch >= epoch_to_init - 1:
                hist4 = torch.mean(hist4, dim = 1)
                knn_base.fit(hist4.detach().cpu().numpy(), labels.cpu().numpy())

        ce_loss.append(train_loss)
        if check_baseline_knn_argument and epoch >= epoch_to_init - 1:
            accuracy_for_knn.append(evaluate_model.knn_baseline_evaluation(student, knn_base, test_loader))
            print('MI is ', calculated_mi)
            print('and MI2 is ', calculated_mi2)
            print('knn so far ', accuracy_for_knn)
            knn_base = KNeighborsClassifier(n_neighbors = 3)

        
        if epoch >= epoch_to_init - 1:
            mi_loss.append(calculated_mi)
            mi_loss2.append(calculated_mi2)
            #mi_loss.append(loss2.data.item())
        mi_hist1.append(calculated_mi_for_histogram_1)
        mi_hist2.append(calculated_mi_for_histogram_2)
        mi_hist3.append(calculated_mi_for_histogram_3)
        mi_hist4.append(calculated_mi_for_histogram_4)

        #if epoch == 75:
        #    torch.save(student.state_dict(), path + "/experiment_" + str(exp_number) + "/model_ep75.pt")
        if epoch > 1 and (epoch % eval_freq == 0 or epoch == epochs - 1):
            evaluate_model.evaluate_model_train_test(student, train_loader_original, test_loader, train_accuracy, test_accuracy)
        if epoch > 20 and epoch % 30 == 0:
            torch.save(student.state_dict(), path + "/experiment_"+str(exp_number) + "/"+str(epoch) + "_model_checkpoint.pt")
       

        print("\nLoss, acc = ", train_loss, correct / total, 'for epoch ', epoch + 1)
        print('mi loss ', calculated_mi)
        if histogram_to_transfer == 5:
            print('mi loss 2 : ', calculated_mi2)
        accuracy_saver.append(correct / total)


    plot_accuracy_saver(accuracy_saver = accuracy_saver, path = path, experiment_number = exp_number)
    plot_accuracies(train_accuracy = train_accuracy, test_accuracy = test_accuracy, path = path, experiment_number = exp_number, epochs = epochs, eval_freq=eval_freq)
    plot_loss(loss = ce_loss, experiment_number = exp_number, path = path, epochs = epochs)
    plot_mi(mi = mi_loss, experiment_number = exp_number, path = path, epochs = epochs, number = 1)
    plot_mi(mi = mi_loss2, experiment_number = exp_number, path = path, epochs = epochs, number = 2)
    
    plot_mi(mi = mi_hist1, experiment_number = exp_number, path = path, epochs = epochs, number = 3)
    plot_mi(mi = mi_hist2, experiment_number = exp_number, path = path, epochs = epochs, number = 4)
    plot_mi(mi = mi_hist3, experiment_number = exp_number, path = path, epochs = epochs, number = 5)
    plot_mi(mi = mi_hist4, experiment_number = exp_number, path = path, epochs = epochs, number = 6)
    return train_accuracy, test_accuracy, accuracy_saver
