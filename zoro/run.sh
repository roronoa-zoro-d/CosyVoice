#! /bin/bash

gpu_id=1

export CUDA_VISIBLE_DEVICES=${gpu_id}

python generate_batch_audio.py ${gpu_id} > logs/log.py.gpu${gpu_id}.txt 2>&1 &