#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:user_module Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
""" NASEASearch """

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR

NUM_CLASSES = 1000


class BasePreNet(nn.Module):
    """
    BasePreNet
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        super(BasePreNet, self).__init__()

    @abstractmethod
    def forward(self, inputs):
        """
        Build model's input macro-architecture.

        :param inputs:
        :return: A tensor - input of the TBS block.
        """
        raise NotImplementedError

    @abstractmethod
    def calc_cout(self):
        """
        Return pre output's dimension.
        """
        raise NotImplementedError


class PreNet(BasePreNet):
    """
    PreNet: This class includes necessary pre processing for classification network.
        It should be implemented by user.
    """
    def __init__(self):
        super(PreNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 64,
                               kernel_size=[7, 7],
                               stride=[2, 2],
                               bias=True,
                               padding=3)
        self.batch_norm = nn.BatchNorm2d(64, eps=1e-5, momentum=0.9)
        self.act = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d([3, 3],
                                     stride=2,
                                     padding=1)

    def forward(self, inputs):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = self.conv0(inputs)
        outputs = self.batch_norm(outputs)
        outputs = self.act(outputs)
        outputs = self.max_pool(outputs)
        return outputs

    def calc_cout(self):
        return 64


class BasePostNet(nn.Module):
    """
    BasePostNet
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        super(BasePostNet, self).__init__()

    @abstractmethod
    def forward(self, inputs):
        """
        Build model's output macro-architecture.

        :param inputs:
        :return: A tensor - model's output.
        """
        raise NotImplementedError

    @abstractmethod
    def calc_cout(self):
        """
        Return number of classes.
        """
        raise NotImplementedError


class PostNet(BasePostNet):
    """
    PostNet: This class includes necessary post processing for classification network.
        It should be implemented by user.
    """
    def __init__(self):
        super(PostNet, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fully_connected = nn.Linear(512, NUM_CLASSES)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.fully_connected(outputs)
        return outputs

    def calc_cout(self):
        return NUM_CLASSES


class UserModuleInterface:
    """
    UserModuleInterface
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

    @abstractmethod
    def dataset_define(self, dataset_dir, is_training):
        """
        Build dataset for nas search

        :param dataset_dir:
        :param is_training:
        :return:  dataset iterator and data num
        """
        raise NotImplementedError

    @abstractmethod
    def loss_func(self, labels, logits):
        """
        Loss Function

        :param labels: GT labels.
        :param logits: logits of network's forward pass.
        :return: A Tensor - loss function's loss
        """
        raise NotImplementedError

    @abstractmethod
    def scheduler_define(self, optimizer, steps_per_epoch):
        """
        Define learning rate update scheduler.

        :param optimizer: the optimizer to update network's parameters.
        :param steps_per_epoch: a Python number, total steps in one epoch.
        :return: A pytorch learning rate scheduler object.
        """
        raise NotImplementedError

    @abstractmethod
    def metrics_func(self, eval_dataloader, eval_function):
        """
        define accuracy function

        :param eval_dataloader: the evaluate dataset loader.
        :param eval_function: the function to evaluate the network's performance.
        :return:
        """
        raise NotImplementedError


class UserModule(UserModuleInterface):
    """
    UserModule: Defines some necessary interfaces for searching.
            It should be implemented by user.
    """
    def __init__(self, epoch, batch_size):
        super(UserModule, self).__init__(epoch, batch_size)
        self.crition = nn.CrossEntropyLoss().cuda()

    def loss_func(self, labels, logits):
        if torch.cuda.is_available():
            labels = labels.cuda()
        return self.crition(logits, labels).sum()

    def dataset_define(self, dataset_dir, is_training):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if is_training:
            dataset = datasets.ImageFolder(dataset_dir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        else:
            dataset = datasets.ImageFolder(dataset_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        return dataset

    def scheduler_define(self, optimizer, steps_per_epoch):
        return ExponentialLR(optimizer, gamma=1.0)

    def metrics_func(self, eval_dataloader, eval_function):
        total = 0
        correct = 0
        for _, (images, labels) in enumerate(eval_dataloader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                output = eval_function(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total <= 0:
            raise Exception('Total eval batch num must be greater than 0!')
        return 1 - correct/total

