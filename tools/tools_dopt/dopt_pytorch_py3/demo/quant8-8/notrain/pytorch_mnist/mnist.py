#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script description:pytorch mnist demo.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
""" mnist """
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    """
    model define
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.seq1 = nn.Sequential(
                        nn.Conv2d(16, 16, 3, padding=1),
                        nn.Conv2d(16, 16, 3, padding=1),
        )
        self.pool = nn.MaxPool2d(2, 2, )
        self.fc1 = nn.Linear(16 * 49, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input_tensor):
        input_tensor = self.pool(F.relu(self.conv1(input_tensor)))
        input_tensor = self.pool(F.relu(self.conv2(input_tensor)))
        input_tensor = self.seq1(input_tensor)

        input_tensor = input_tensor.view(-1, 16*49)
        input_tensor = F.relu(self.fc1(input_tensor))
        input_tensor = F.relu(self.fc2(input_tensor))
        input_tensor = self.fc3(input_tensor)
        return input_tensor


def model_define():
    """
    return model
    """
    return Net()

