#!/bin/bash
# Script description:run_develop Script.
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
set -e
clear
mkdir -p log_classification

LOG_FILE="log_classification/log_$(date '+%Y_%m_%d_%H_%M_%S').txt"

ROOT=$(cd ../../../; pwd)
target_dir="${ROOT}"/dopt/strategy/task/task_custom/nas_ea/pretrain
cp -r user_module.py "${target_dir}"
if [ -f "${ROOT}"/dopt/constraints/latency_predictor/models/table.tar.gz ];then
  tar -xzf "${ROOT}"/dopt/constraints/latency_predictor/models/table.tar.gz -C "${ROOT}"/dopt/constraints/latency_predictor/models/
  rm -f "${ROOT}"/dopt/constraints/latency_predictor/models/table.tar.gz
fi

python3 "${ROOT}"/dopt_so.py -c scen.yaml 2>&1 | tee ./"${LOG_FILE}"
