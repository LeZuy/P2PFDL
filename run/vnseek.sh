#!/usr/bin/env bash

# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True # ROCm 6.2: slow-down program
# export PYTORCH_NO_HIP_MEMORY_CACHING=1 # ROCm 6.2: slow-down program
# export HSA_ENABLE_SDMA=0 # ROCm 6.2: slow-down program
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export HIP_VISIBLE_DEVICES=0,1

run_folder="./run/"
log_file="${run_folder}/vnseek.log"
err_file="${run_folder}/vnseek.err"

source venv_rocm62/bin/activate
# python3.10 -m cProfile -o profile.out main.py --config configs/experiment.yaml > ${log_file} 2> ${err_file}
python3.10 main.py --config configs/experiment.yaml > ${log_file} 2> ${err_file}