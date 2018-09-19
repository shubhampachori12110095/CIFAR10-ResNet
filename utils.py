import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_data_loaders():

    transform = {
        'train': transforms.Compose([

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

    dataset = {data_set: torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform[data_set])
        for data_set in ['train', 'val']}

    val_set_size = 0.2

    split = int(np.floor(val_set_size * len(dataset['train'])))

    indices = list(range(len(dataset['train'])))

    np.random.seed()  # add seed to set random
    np.random.shuffle(indices)

    train_i, val_i = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_i)
    val_sampler = SubsetRandomSampler(val_i)

    data_loaders = {d_set: torch.utils.data.DataLoader(
        dataset[d_set], batch_size=10, sampler=train_sampler if d_set == 'train' else val_sampler, num_workers=4)
        for d_set in ['train', 'val']}

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform['test'])

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    return data_loaders, test_loader

