# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan 29 15:21:27 2024

# @author: demi
# """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from torch.autograd import Variable
from tddl.factorizations import factorize_network
from tddl.factorizations import number_layers


# class GaripovNet(nn.Module):
#     def __init__(
#         self, in_channels, num_classes, conv1_out=64, conv2_out=64, conv3_out=128, conv4_out=128, conv5_out=128, conv6_out=128, fc_in=128,
#     ):
#         super(GaripovNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, conv1_out, 3, 1, padding=1)
#         self.conv1_bn = nn.BatchNorm2d(conv1_out)
#         self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, 1, padding=1)
#         self.conv2_bn = nn.BatchNorm2d(conv2_out)
#         self.conv3 = nn.Conv2d(conv2_out, conv3_out, 3, 1, padding=1)
#         self.conv3_bn = nn.BatchNorm2d(conv3_out)
#         self.conv4 = nn.Conv2d(conv3_out, conv4_out, 3, 1, padding=1)
#         self.conv4_bn = nn.BatchNorm2d(conv4_out)
#         self.conv5 = nn.Conv2d(conv4_out, conv5_out, 3, 1, padding=1)
#         self.conv5_bn = nn.BatchNorm2d(conv5_out)
#         self.conv6 = nn.Conv2d(conv5_out, conv6_out, 3, 1, padding=1)
#         self.fc1 = nn.Linear(fc_in, num_classes) # TODO: change 12 by 12 # TODO: check num_classes
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = F.relu(x)

#         x = self.conv2(x)
#         x = self.conv2_bn(x)
#         x = F.relu(x)o
#         x = F.max_pool2d(x, kernel_size=3, stride=2)

#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = F.relu(x)

#         x = self.conv4(x)
#         x = self.conv4_bn(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=3, stride=2)

#         x = self.conv5(x)
#         x = self.conv5_bn(x)
#         x = F.relu(x)

#         x = self.conv6(x)
#         x = F.avg_pool2d(x, kernel_size=4)
        
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
        
#         output = F.log_softmax(x, dim=1) # TODO: check if needed
#         return output
# model = GaripovNet(*args, **kwargs)

# model.load_state_dict(torch.load("/home/demi/Documents/tddl/pretrained/f_mnist/logs/garipov/baselines/1647955843/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue"))



class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_f_mnist_loader(path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.FashionMNIST(path, train=True, download=True)
    train_dataset, valid_dataset = random_split(
        dataset,
        (50000, 10000),
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset = DatasetFromSubset(
        train_dataset, transform=transform_train,
    )
    valid_dataset = DatasetFromSubset(
        valid_dataset, transform=transform_test,
    )

    test_dataset = datasets.FashionMNIST(path, train=False, transform=transform_test)
    
    return train_dataset, valid_dataset, test_dataset


def fmnist_stratified_loaders(
    path: Path,
    batch_size: int,
    data_workers: int,
    valid_size: int = 5000,
    random_transform_training: bool = True,
) -> Tuple[DataLoader,...]:
    '''
        input:
        - valid_size: total number of validation observations
    '''
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]) if random_transform_training else transform_test

    dataset = datasets.FashionMNIST(path, train=True, download=True)

    num_train = len(dataset)
    indices = list(range(num_train))

    train_idx, valid_idx, _, _ = train_test_split(indices, 
        dataset.targets, test_size=valid_size, 
        stratify=dataset.targets, random_state=42,
    )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataset = datasets.FashionMNIST(
        root=path, train=True,
        download=True, transform=transform_train,
    )

    valid_dataset = datasets.FashionMNIST(
        root=path, train=True,
        download=True, transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=data_workers,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, 
        sampler=valid_sampler,
        num_workers=data_workers,
    )

    test_dataset = datasets.FashionMNIST(
        path, train=False, transform=transform_test, download=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=data_workers,
    )

    return train_loader, valid_loader, test_loader


model=torch.load("/home/demi/Documents/tddl/pretrained/f_mnist/logs/garipov/baselines/1647955843/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue/cnn_best.pth")

# #extract data
# transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5),)])
# trainset=datasets.FashionMNIST('~/.pytorch/F_MMINST_data', download=True, train=True, transform=transform)
# testset=datasets.FashionMNIST('~/.pytorch/F_MMINST_data', download=True, train=False, transform=transform)

# indices=list(range(len(trainset)))
# np.random.shuffle(indices)

# split=int(np.floor(0.2* len(trainset)))
# train_sample=SubsetRandomSampler(indices[:split])
# valid_sample=SubsetRandomSampler(indices[split:])

# #dataloaders
# trainloader=torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64)
# validloader=torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64)
# testloader=torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# #testing the model
# test_loss=0
# class_correct=list(0. for i in range(10))
# class_total=list(0. for i in range(10))

# model.eval()
# criterion=nn.NLLLoss()
# for images,labels in testloader:
#     #forward pass
#     output=model(images)
#     #calculate the loss
#     loss=criterion(output,labels)
#     #update test loss
#     test_loss+=loss.item()*images.size(0)
#     #convert output probibilities to predicted class
#     _, pred=torch.max(output,1)
#     #compare predictions to true lables
#     correct=np.squeeze(pred.eq(labels.data.view_as(pred)))
    
#     #calculate test accuract for each object class
#     for i in range(len(labels)):
#         label=labels.data[i]
#         class_correct[label]+=correct[i].item
#         class_total[label] +=1


# test_loss=test_loss/len(testloader.sampler)

# print('Test loss: {:.6f}\n'.format(test_loss))
trainloader, validloader, testloader=fmnist_stratified_loaders(path=" ~/.pytorch/F_MMINST_data", batch_size=256, data_workers=2, valid_size=5000)

model.cuda()
model.eval()

correct = 0
steps = 0
total_time = 0
val_loss = 0.0

t = tqdm(testloader, total=int(len(testloader)))
criterion=torch.nn.CrossEntropyLoss()

for i, (batch, label) in enumerate(t):
            with torch.no_grad():
                batch = batch.cuda() # TODO: change to .to_device(device)
                # t0 = time.time()
                # input = Variable(batch)
                output = model(Variable(batch)) #.cpu()  # commented cpu() out
                loss = criterion(output, Variable(label, requires_grad=False).cuda())
                val_loss += loss.cpu().numpy()
                # t1 = time.time()
                # total_time = total_time + (t1 - t0)
                pred = output.cpu().data.max(1)[1] # added .cpu()
                correct += pred.cpu().eq(label).sum() # TODO check if output.cpu() and pred.cpu() is necessary
                steps += label.size(0)

model.train()
accuracy = float(correct) / steps
val_loss=val_loss/steps
print(val_loss)
print(accuracy)


factorize_network(      
    model,                  # Changes your pytorch model inplace.
    layers=[2,4],                # Modifies only layers (or layer type) you specify,
    factorization='cp', # into specific factorization,
    rank=0.5,               # with a given (fractional) rank.
    decompose_weights=True, # Decompose the weights of the model.
)

model.cuda()
model.eval()

correct = 0
steps = 0
total_time = 0
val_loss = 0.0

t = tqdm(testloader, total=int(len(testloader)))
criterion=torch.nn.CrossEntropyLoss()

for i, (batch, label) in enumerate(t):
            with torch.no_grad():
                batch = batch.cuda() # TODO: change to .to_device(device)
                # t0 = time.time()
                # input = Variable(batch)
                output = model(Variable(batch)) #.cpu()  # commented cpu() out
                loss = criterion(output, Variable(label, requires_grad=False).cuda())
                val_loss += loss.cpu().numpy()
                # t1 = time.time()
                # total_time = total_time + (t1 - t0)
                pred = output.cpu().data.max(1)[1] # added .cpu()
                correct += pred.cpu().eq(label).sum() # TODO check if output.cpu() and pred.cpu() is necessary
                steps += label.size(0)

model.train()
accuracy = float(correct) / steps
val_loss=val_loss/steps
print(val_loss)
print(accuracy)




