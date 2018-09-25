import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os


def get_data_loaders():

    transform = {
        'train': transforms.Compose([

            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        ]),

        'val': transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        ]),

        'test': transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        ])
    }

    data_set = {data_set: torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform[data_set])
        for data_set in ['train', 'val']}

    val_set_size = 0.2

    split = int(np.floor(val_set_size * len(data_set['train'])))

    indices = list(range(len(data_set['train'])))

    np.random.seed()  # add seed to set random
    np.random.shuffle(indices)

    train_i, val_i = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_i)
    val_sampler = SubsetRandomSampler(val_i)

    data_loaders = {d_set: torch.utils.data.DataLoader(
        data_set[d_set], batch_size=10, sampler=train_sampler if d_set == 'train' else val_sampler, num_workers=4)
        for d_set in ['train', 'val']}

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform['test'])

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    return data_loaders, test_loader


def load_checkpoint(model, optimizer, scheduler, filename):

    if os.path.isfile(filename):

        checkpoint = torch.load(filename)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        
        return epoch

    else:

        print('File Not Found')
