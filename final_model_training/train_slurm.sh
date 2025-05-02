:'
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'

#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_carbonCLIP
#SBATCH --time=24:00:00

module load anaconda3
source activate env

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=13805

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
dataset_dir="" #set dataset directory

cd ..

    srun --cpu_bind=v --accel-bind=gn python -u $HOME/carbon-nas/open_clip_custom/src/open_clip_train/main.py \
    --train-data=$dataset_dir \
    --batch-size=128 \
    --epochs=2 \
    --workers=8 \
    --model ViT-B-16 \
    --pretrained datacomp_xl_s13b_b90k \
    --name "carbonCLIP" \
    --lr 5e-4 \
    --wd 0.2  \
    --warmup 2000 \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --text-layers 7 --text-embed-dim 512 --text-ffn-dim 2048 --text-head-num 6 \
    --vision-layers 11 --vision-embed-dim 768 --vision-ffn-dim 3072 --vision-head-num 10  \
    --train-num-samples 2500000000 \
    --distill-model=ViT-B-16   --distill-pretrained=datacomp_xl_s13b_b90k


