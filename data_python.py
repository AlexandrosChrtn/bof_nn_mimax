import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data.dataloader

def cifar10_loader(data_path='../data', batch_size=128, augment_train = False):
    """
    Loads the cifar10 dataset in torch-ready format
    :param data_path:
    :param batch_size:
    :return:
    """

    if augment_train:
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    

    train_data = dset.CIFAR10(data_path, train=True, transform=train_transform, download=True)
    train_data_original = dset.CIFAR10(data_path, train=True, transform=test_transform, download=True)
    test_data = dset.CIFAR10(data_path, train=False, transform=test_transform, download=True)

    sample = list(range(1, len(train_data), 2))
    train_data_sample = torch.utils.data.Subset(train_data_original, sample)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2,
                                              pin_memory=True)
    train_subset_loader = torch.utils.data.DataLoader(train_data_sample, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=100,
                                                        shuffle=True, num_workers=2, pin_memory=True)
    bof_center_loader = torch.utils.data.DataLoader(train_data_original, batch_size=500,
                                                        shuffle=True, num_workers=2, pin_memory=True)
    return train_loader, test_loader, train_loader_original, train_subset_loader, bof_center_loader

def mnist_loader(data_path='../data', batch_size=128):
    """
    Loads the mnist dataset in torch-ready format
    :param data_path:
    :param batch_size:
    :return:
    """
    mean = [x / 255 for x in [125.3]]
    std = [x / 255 for x in [63.0]]

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = dset.MNIST(data_path, train=True, transform=train_transform, download=True)
    train_data_original = dset.MNIST(data_path, train=True, transform=test_transform, download=True)
    test_data = dset.MNIST(data_path, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                              pin_memory=True)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=batch_size,
                                                        shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_loader_original