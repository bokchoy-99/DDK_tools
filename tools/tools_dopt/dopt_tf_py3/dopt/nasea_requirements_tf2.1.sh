#!/bin/bash
# encoding: utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
set -e
clear

for line in $(cat $(dirname $0)/strategy/task/task_custom/nas_ea/requirements_tf2.1.txt)
do
    pip3 install $line
done
