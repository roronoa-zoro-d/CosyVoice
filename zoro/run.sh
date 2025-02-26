#! /bin/bash


if [ $# -ne 1 ];then
    echo "Usage:  <gpu_id>"
    exit 1
fi

gpu_id=$1
echo "gpu_id: ${gpu_id}"

exit

export CUDA_VISIBLE_DEVICES=${gpu_id}

python generate_batch_audio.py ${gpu_id} > logs/log.py.gpu${gpu_id}.txt 2>&1 &