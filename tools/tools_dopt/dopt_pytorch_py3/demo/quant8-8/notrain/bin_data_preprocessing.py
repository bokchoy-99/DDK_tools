#!/usr/bin/env python
# encoding: utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
""" data preprocessing """
import os
import numpy as np


def dump_tensor(tensor, tensor_shape, output_path):
    """
    dump 4D-tensor, input format: NCHW
    """
    magic_number = 510
    output_path = os.path.realpath(output_path)
    with os.fdopen(os.open(output_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC,
                           0o700), 'wb') as output_file:
        output_file.write(int.to_bytes(magic_number, \
                          length=4, byteorder='little'))
        for tensor_str in tensor_shape:
            output_file.write(int.to_bytes(tensor_str, \
                              length=4, byteorder='little'))
        output_file.write(tensor.tobytes())


def dump_flexible_tensor(tensor, tensor_shape, output_path):
    """
    dump non-4D-tensor
    """
    rank = len(tensor_shape)
    magic_number = 610
    with os.fdopen(
        os.open(
            output_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o700
        ), 'wb'
    ) as output_file:
        output_file.write(int.to_bytes(magic_number, length=4, \
                          byteorder='little'))
        output_file.write(int.to_bytes(rank, length=4, byteorder='little'))
        for tensor_str in tensor_shape:
            output_file.write(int.to_bytes(tensor_str, length=4, \
                              byteorder='little'))
        output_file.write(tensor.tobytes())


def tensor_preprocessing(shape_str, tensor, output_path):
    """
    data preprocessing with different dimensions
    """
    shape_str = shape_str.split(',')
    tensor_shape = [int(s) for s in shape_str]
    if len(tensor_shape) == 4:
        dump_tensor(tensor, tensor_shape, output_path)
    else:
        dump_flexible_tensor(tensor, tensor_shape, output_path)

if __name__ == "__main__":

    # Attention: set your own calibration data as 'tensor'
    shape_info = "N,3,224,224"
    N = 1
    input_tensor = np.random.rand(N*3*224*224).astype(np.float32)
    file_path = "./output.bin"

    tensor_preprocessing(shape_info, input_tensor, file_path)

