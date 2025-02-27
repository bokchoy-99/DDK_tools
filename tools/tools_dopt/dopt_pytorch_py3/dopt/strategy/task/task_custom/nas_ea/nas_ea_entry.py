#!/usr/bin/env python
# coding: utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
"""NASEASearch """
import argparse
from user_module import PreNet
from user_module import PostNet
from user_module import UserModule
from nas_ea import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NASEASearch')
    parser.add_argument('--cfg_path', metavar='DIR', help='path to config file')
    args = parser.parse_args()
    main(args, PreNet, PostNet, UserModule)

