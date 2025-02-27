#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:user_module Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
""" NASEASearch """

from collections import OrderedDict
from abc import ABCMeta
from abc import abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torchvision

from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models.segmentation.fcn import FCNHead

import references.segmentation.transforms as T
from references.segmentation.utils import ConfusionMatrix

NUM_CLASSES = 21


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


def get_transform(train):
    """
    build torchvision transformations.
    :param train: boolean, training mode or not.
    :return: torchvision transforms instance.
    """
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.CenterCrop(crop_size))
    transforms.append(ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


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
    PostNet: This class includes necessary pre processing for segmentation network.
            It should be implemented by user.
    """

    def __init__(self):
        super(PreNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=[2, 2], bias=True, padding=3)
        self.batch_norm = nn.BatchNorm2d(64, eps=1e-5, momentum=0.9)
        self.act = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d([3, 3], stride=2, padding=1)

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
    PostNet: This class includes necessary post processing for segmentation network.
            It should be implemented by user.
    """

    def __init__(self):
        super(PostNet, self).__init__()
        atrous_rates = [6, 12, 18]
        inplanes = 512
        num_classes = NUM_CLASSES
        self.input_shape = [480, 480]

        self.classifier = DeepLabHead(inplanes, num_classes, atrous_rates)
        self.aux_classifier = FCNHead(inplanes // 2, num_classes)

    def forward(self, inputs):
        result = OrderedDict()
        output = inputs[1]
        output = self.classifier(output)
        output = F.interpolate(output, size=self.input_shape, mode='bilinear',
                               align_corners=False)
        result["out"] = output

        if self.aux_classifier is not None:
            aux_output = inputs[0]
            aux_output = self.aux_classifier(aux_output)
            aux_output = F.interpolate(aux_output, size=self.input_shape, mode='bilinear',
                                       align_corners=False)
            result["aux"] = aux_output
        return result

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
        self.num_classes = 21

    def loss_func(self, labels, logits):
        losses = {}
        if torch.cuda.is_available():
            labels = labels.cuda()
        for name, pred in logits.items():
            losses[name] = nn.functional.cross_entropy(pred, labels,
                                                       ignore_index=255)
        if len(losses) == 1:
            return losses['out']
        return losses['out'] + 0.5 * losses['aux']

    def dataset_define(self, dataset_dir, is_training):
        ds_fn = torchvision.datasets.VOCSegmentation
        if is_training:
            dataset = ds_fn(dataset_dir, image_set="train",
                            transforms=get_transform(train=True))
        else:
            dataset = ds_fn(dataset_dir, image_set="val",
                            transforms=get_transform(train=False))
        return dataset

    def scheduler_define(self, optimizer, steps_per_epoch):
        if steps_per_epoch == 0 or self.epoch == 0:
            raise Exception('Steps_per_epoch and Epoch must be non-zero!')
        lr_lambda = lambda x: (1 - x / (steps_per_epoch * self.epoch)) ** 0.9
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def metrics_func(self, eval_dataloader, eval_function):
        conf_mat = ConfusionMatrix(self.num_classes)
        with torch.no_grad():
            for _, (images, labels) in enumerate(eval_dataloader):
                if torch.cuda.is_available():
                    labels = labels.cuda()
                output = eval_function(images)
                output = output['out']
                conf_mat.update(labels.flatten(), output.argmax(1).flatten())
            conf_mat.reduce_from_all_processes()

        _, _, iou = conf_mat.compute()
        mean_iou = iou.mean().item()
        return 1 - mean_iou


class DeepLabHead(nn.Sequential):
    """
    Deeplab head part.
    """
    def __init__(self, in_channels, num_classes, atrous_rates):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

