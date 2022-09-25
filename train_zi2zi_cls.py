# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
from logging import lastResort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as nph
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib
# from test_embedded import Get_test_results_single, normalize
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
import random
import time
import os
from model import ft_zi2zi_Gen
import json
# import visdom
import copy

######################################################################
# Options
root = '/home2/prof/ro/team9' #uosb
mode = 'uosbigdata'
# mode = 'sdsserver'

if mode == 'sdsserver':
    root = '/home/roh/Documents/team9'


class NegSamGenerator(object):

    def __init__(self, ns_range):
       self.min = ns_range[0]
       self.max = ns_range[1]

    def __call__(self, sample):

        while True:
            middle1 = random.randrange(10, 90, 1) * 0.01
            middle2 = random.randrange(10, 90, 1) * 0.01
            if middle2 > middle1:
                break

        difference = int(random.randrange(self.min, self.max, 1))  # *0.01*input_image.shape[2])
        direction = random.randrange(0, 2, 1)  # 0 is upward 1 is downward

        nagative_image = copy.deepcopy(sample)
        temp = nagative_image[:, :, int(sample.shape[2] * middle1):int(sample.shape[2] * middle2)]
        if direction == 0:  # upward
            temp = torch.cat((temp[:, difference:, :], temp[:, :difference, :]), dim=1)
        else:
            temp = torch.cat((temp[:, -difference:, :], temp[:, :-difference, :]), dim=1)

        nagative_image[:, :, int(sample.shape[2] * middle1):int(sample.shape[2] * middle2)] = temp

        return nagative_image


class Padtofixedsize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, sample):
        sample = torch.squeeze(sample,dim=0)
        pad = (0,(self.size - sample.shape[1]))
        sample = F.pad(sample, pad, "constant", 0 )
        return torch.unsqueeze(sample,dim=0)


def visual_sample(samples, savefile=True):
    cnt = 0
    for sample_image in samples:
        cnt += 1
        sample_image = sample_image.squeeze()
        sample_image = sample_image * 0.5 + 0.5
        plt.imshow(sample_image.cpu(), cmap='gray')
        if savefile:
            plt.savefig(f'sample_{cnt}.png')
        else:
            plt.show()


def softmax(x):
    # x = x.exp()/x.exp().sum(dim=1)[:, None]
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim=1)
    out = x_exp / x_exp_sum.unsqueeze(dim=1)
    return out


def cross_entropy(x, y):
    # eps = 1e-7
    # x = x + eps
    loss = -torch.log(x) * y
    loss = loss.sum() / x.shape[0]  # size(dim=0)
    return loss


def train_model(model, batch_size, opt, criterion, dataloader, optimizer, scheduler, set_name, use_gpu=False,
                num_epochs=25):
    print_mode = True

    since = time.time()
    train_losses = []
    test_losses = []
    pos_d_acc = []
    neg_d_acc = []
    test_acc = []
    test1_acc = []
    test2_acc = []

    ns_range = [opt.min_ns, opt.max_ns]
    print_cut = 100

    for epoch in range(num_epochs):
        if print_mode:
            print('Epoch {}/{} --------------------------------'.format(epoch, num_epochs))
        # Each epoch has a training and validation phase
        # zero is original, one is fake image
        for phase in ['train', 'test', 'test1', 'test2']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                labels = torch.cat((torch.zeros(batch_size[phase][0], dtype=torch.int64),torch.ones(batch_size[phase][1], dtype=torch.int64)),dim=0)
            elif phase == 'test':
                model.train(False)  # Set model to evaluate mode
                labels = torch.zeros(batch_size[phase][0], dtype=torch.int64)
                label_index = 0
            elif phase == 'test1':
                model.train(False)  # Set model to evaluate mode
                labels = torch.ones(batch_size[phase][0], dtype=torch.int64)
                label_index = 1
            else: 
                model.train(False)  # Set model to evaluate mode
                labels = torch.ones(batch_size[phase][0], dtype=torch.int64)
                label_index = 1

            running_norm = []
            running_loss = 0.0
            running_corrects = 0
            running_positive_corrects = 0
            running_negative_corrects = 0
            train_batch = 0
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = torch.Tensor()
            if use_gpu:
                outputs = outputs.cuda()
            
            for iter_ in range(len(dataloader[phase][0])//batch_size[phase][0]): # train에서는 batch size로 나눈 만큼 itration 반복, test는 1 iteration

                for k in range(len(dataloader[phase])):
                    for i, data in enumerate(dataloader[phase][k]):
                        if use_gpu:
                            inputs = data[0].cuda()

                        temp = model(inputs)
                        outputs = torch.cat((outputs, temp), dim=0)
                        if i == batch_size[phase][k]-1:
                            break

                if use_gpu: 
                    labels = labels.cuda()
                _, preds = torch.max(outputs, 1)
                #outputs = softmax(outputs)
                loss = criterion(outputs, labels)

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    outputs = torch.Tensor() #ouputs
                    if use_gpu:
                        outputs = outputs.cuda()

                    running_positive_corrects += torch.sum(preds[:batch_size[phase][0]] == labels[:batch_size[phase][0]].data)  
                    running_negative_corrects += torch.sum(preds[batch_size[phase][0]:] == labels[batch_size[phase][0]:].data) 

            if phase == 'train':
                scheduler.step()

            running_corrects = running_corrects.float()

            total_batch_size = 0
            for k in range(len(batch_size[phase])):
                total_batch_size += batch_size[phase][k]
            epoch_loss = running_loss / ((iter_+1)* total_batch_size)
            epoch_acc = running_corrects / ((iter_+1)* total_batch_size)

            if phase == 'train':
                epoch_loss = running_loss / ((iter_+1)* total_batch_size)
                epoch_acc = running_corrects / ((iter_+1)*  total_batch_size)  
                epoch_positive_acc = running_positive_corrects / ((iter_+1)* batch_size[phase][0])
                epoch_negative_acc = running_negative_corrects / ((iter_+1)* batch_size[phase][1])
                pos_d_acc.append(epoch_positive_acc.cpu())
                neg_d_acc.append(epoch_negative_acc.cpu())

                if epoch > 95:
                    save_network(model, path, epoch)
                if print_mode:
                    print(f'{phase} Loss: {epoch_loss:.8f} Acc: {epoch_acc:.8f} Postive_Acc: {epoch_positive_acc:.4f} Negative_Acc: {epoch_negative_acc:.4f}')

            else:
                prob = softmax(outputs)
                prob = prob.cpu().detach().numpy()
                prob = prob[:, label_index].round(2)
                if print_mode:
                    print(prob)
                    print(f'{set_name}: {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:4f}')

                if phase == 'test':
                    test_acc.append(epoch_acc.cpu())
                if phase == 'test1':
                    test1_acc.append(epoch_acc.cpu())
                if phase == 'test2':
                    test2_acc.append(epoch_acc.cpu())
                    x = [x for x in range(1, len(test_acc) + 1)]
                    plt.plot(x, pos_d_acc, x, neg_d_acc, x, test_acc, x, test1_acc, x, test2_acc)
                    plt.legend(['Pos_Acc', 'Neg_Acc', 'Test_Acc', 'Test1_Acc', 'Test2_Acc'])
                    plt.savefig(f'./n116figures4/Acc_{set_name}.png')
                    plt.clf()

    # time_elapsed = time.time() - since    
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))

    return model


####################################################
# Save model
# ---------------------------
def save_network(model, path, epoch_label):
    save_filename = f'net_{epoch_label}.pth'
    save_path = os.path.join(path, save_filename)
    torch.save(model.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        model = model.cuda()
######################################################################


def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network


# if not os.path.isdir(dir_name):``
#     os.mkdir(dir_name)

if __name__ == '__main__':
    data_dir = f'{root}/Dataset_date_renamed_add_eng'
    # data_dir = f'{root}/team9_traindata_split20'

    use_gpu = True

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default=0, type=int, help='gpu_ids: 0  1')
    parser.add_argument('--name', default='d116_test_no_aug', type=str, help='output model name')
    parser.add_argument('--batchsize', default=10, type=int, help='batchsize')
    parser.add_argument('--min_ns', default=6, type=int, help='min_ns')
    parser.add_argument('--max_ns', default=12, type=int, help='max_ns')
    parser.add_argument('--tl', default=0.05, type=float, help='translate')
    parser.add_argument('--neg_balance', default=1, type=float, help='negative sample ratio')
    parser.add_argument('--norm', default=True, type=bool, help='norm_feature')
    parser.add_argument('--blur', default=0.2, type=float, help='Gaussian blur')
    parser.add_argument('--epoch_sch', default=(50,80,100), type=tuple, help='train in epoch schedule')


    opt = parser.parse_args()

    if use_gpu:
        torch.cuda.set_device(opt.gpu_ids)
    model_path = f'{root}/ft_zi2zi_gen3.pth'
    lr5 = 0.00001
    lr4 = 0.0001
    lr3 = 0.001
    lr2 = 0.01
    lr1 = 0.1
    e_drop = opt.epoch_sch[0]
    e_mid = opt.epoch_sch[1]
    e_end = opt.epoch_sch[2]
    train_batch_size = opt.batchsize 
    
    
    version = 0
    set_name = f'{opt.name}_ns{opt.min_ns}_{opt.max_ns}_nz{opt.norm}_tl{opt.tl}_b{opt.batchsize}_nb_{opt.neg_balance}_bl{opt.blur}_v{version}'

    
    path = f'{root}/trained_{set_name}/'
    while os.path.isdir(path):
        version += 1
        set_name = f'{opt.name}_ns{opt.min_ns}_{opt.max_ns}_nz{opt.norm}_tl{opt.tl}_b{opt.batchsize}_nb_{opt.neg_balance}_bl{opt.blur}_v{version}'
        path = f'{root}/trained_{set_name}/'
            
    os.mkdir(path)

    # test_batch_size = 34
    ######################################################################
    # Load Data
    p_blur = opt.blur

    resize = (80,)
    train_positive_T_list = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomApply(transforms=[transforms.RandomAffine(degrees=0, translate=(opt.tl,opt.tl),fill=255)], p=1),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            #Padtofixedsize(500),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(3,3))], p=p_blur),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(5,5))], p=p_blur),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(7,7))], p=p_blur),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(9,9))], p=p_blur),
        ])

    train_negative_T_list = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomApply(transforms=[transforms.RandomAffine(degrees=0, translate=(opt.tl,opt.tl),fill=255)], p=1),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            #Padtofixedsize(500),
            NegSamGenerator(ns_range=[opt.min_ns, opt.max_ns]),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(3,3))], p=p_blur),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(5,5))], p=p_blur),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(7,7))], p=p_blur),
            transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(9,9))], p=p_blur),
        ])

    test_transform_list = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            # transforms.RandomHorizontalFlip(p=0.5)
        ])
    
    image_datasets_train_pos = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_positive_T_list)
    image_datasets_train_neg = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_negative_T_list)

    image_datasets_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transform_list)
    image_datasets_test1 = datasets.ImageFolder(os.path.join(data_dir, 'test1'), test_transform_list)
    image_datasets_test2 = datasets.ImageFolder(os.path.join(data_dir, 'test2'), test_transform_list) # new 


    dataloaders = {}
    dataloaders['train'] = []
    dataloaders['train'].append(torch.utils.data.DataLoader(image_datasets_train_pos, batch_size=1, shuffle=True))
    dataloaders['train'].append(torch.utils.data.DataLoader(image_datasets_train_neg, batch_size=1, shuffle=True))

    dataloaders['test'] = [] 
    dataloaders['test'].append(torch.utils.data.DataLoader(image_datasets_test, batch_size=1, shuffle=False))
    dataloaders['test1'] = [] 
    dataloaders['test1'].append(torch.utils.data.DataLoader(image_datasets_test1, batch_size=1, shuffle=False))
    dataloaders['test2'] = [] 
    dataloaders['test2'].append(torch.utils.data.DataLoader(image_datasets_test2, batch_size=1, shuffle=False))

    batch_size = {}
    batch_size['train'] = []
    batch_size['train'].append(train_batch_size)
    batch_size['train'].append(int(train_batch_size*opt.neg_balance))

    batch_size['test'] = []
    batch_size['test1'] = []
    batch_size['test2'] = []
    batch_size['test'].append(int(len(dataloaders['test'][0])))
    batch_size['test1'].append(int(len(dataloaders['test1'][0])))
    batch_size['test2'].append(int(len(dataloaders['test2'][0])))

    model = ft_zi2zi_Gen()
    # model = resdual18(10)
    model = load_network_path(model, model_path)
    model.to_InstanceNorm()
    if opt.norm:
        model.set_norm()
    #model.set_classifier()
    #model.set_BatchNorm(0.0001)

    if use_gpu:
        model = model.cuda()
        # nn.DataParallel(model, device_ids=[2,3]).cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()


    params_ft = []
    params_ft.append({'params': model.layer1.parameters(), 'lr': lr5})  # 0.00001
    params_ft.append({'params': model.layer2.parameters(), 'lr': lr5})
    params_ft.append({'params': model.layer3.parameters(), 'lr': lr5})
    params_ft.append({'params': model.layer4.parameters(), 'lr': lr4})
    params_ft.append({'params': model.layer5.parameters(), 'lr': lr4})
    params_ft.append({'params': model.layer6.parameters(), 'lr': lr3})
    params_ft.append({'params': model.layer7.parameters(), 'lr': lr2})
    params_ft.append({'params': model.layer8.parameters(), 'lr': lr2})
    params_ft.append({'params': model.classifier.parameters(), 'lr': lr2})

    # params_ft = []
    # params_ft.append({'params': model.layer1.parameters(), 'lr': lr3})
    # params_ft.append({'params': model.layer2.parameters(), 'lr': lr3})
    # params_ft.append({'params': model.layer3.parameters(), 'lr': lr2})
    # params_ft.append({'params': model.layer4.parameters(), 'lr': lr2})
    # params_ft.append({'params': model.layer5.parameters(), 'lr': lr2})
    # params_ft.append({'params': model.layer6.parameters(), 'lr': lr2})
    # params_ft.append({'params': model.layer7.parameters(), 'lr': lr1})
    # params_ft.append({'params': model.layer8.parameters(), 'lr': lr1})
    # # learning rate should be applied to classifier as well
    # params_ft.append({'params': model.classifier.parameters(), 'lr': lr2})

    #optimizer_ft = optim.SGD(params_ft, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer_ft = optim.Adam(params_ft, betas=(0.5, 0.999))

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=e_drop, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[e_drop, e_mid], gamma=0.1)
    print(f'start to {set_name} ver:{version}')
    lrs = ''
    for k in range(len(params_ft)):
        lrs += f"{round(params_ft[k]['lr'], 6)},"

    model = train_model(model, batch_size, opt, criterion, dataloaders, optimizer_ft, exp_lr_scheduler, set_name,
                        use_gpu, num_epochs=e_end)

    print(lrs)
    print(f'This trial name is :{set_name}, epoch schedule: {e_drop}, {e_mid}, {e_end}')
