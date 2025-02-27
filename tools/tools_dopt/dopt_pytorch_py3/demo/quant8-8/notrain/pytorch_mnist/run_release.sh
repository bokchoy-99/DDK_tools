#!/bin/bash
# Scipt function desciption:tensorflow_imagenet_resnet50 run_release Scipt.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
set -e
clear

# replace the root path in template files
ROOT=$(cd ../../../..; pwd)
find -L . -name "*.tmp" | xargs -i -n1 -r sh -c 'FN_TMP={}; FN=${FN_TMP%.*}; cp ${FN_TMP} ${FN}; sed -i -e "s#@ROOT@#'"${ROOT}"'#g" ${FN}'

python ${ROOT}/dopt_so.py  --framework 5 -m   0 --weight mnist.pth --model mnist.py --cal_conf ./config.prototxt --output  ./mnist_quant.onnx   --input_shape   x:1,1,28,28  --compress_conf  ./mnist_param --device_idx   0

