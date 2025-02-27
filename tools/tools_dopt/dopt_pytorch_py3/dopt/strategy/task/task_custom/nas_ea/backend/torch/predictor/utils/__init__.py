#!/usr/bin/env python
# coding: utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
"""NASEASearch """


def encode_one_hot(block_choice, num_blocks):
    """

    :param block_choice:
    :return:
    """
    layernum = len(block_choice)
    encode_list = []
    block = [num_blocks] * layernum
    for i in range(layernum):
        encode_list.append([])
        for _ in range(block[i]):
            encode_list[i].append(0)

    for layer, index in enumerate(block_choice):
        encode_list[layer][index] = 1
    return encode_list

