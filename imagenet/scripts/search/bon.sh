#!/bin/bash

## annotated params are for target 444

export CUDA_VISIBLE_DEVICES=6,

## Parameters
data_type=image
image_size=256
dataset="imagenet"
model_name_or_path='models/openai_imagenet.pt'

task=label_guidance
guide_network='google/vit-base-patch16-224'
global_verifier="google/vit-base-patch16-384"
target=222
# target=444

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=12    ## number of particles
num_samples=256
logging_resolution=512
guidance_name='bfs'
eval_batch_size=32
wandb=False

rho=0.2
mu=0.4
sigma=0.1
# rho=0.2
# mu=0.8
# sigma=0.01
eps_bsz=1
iter_steps=4

start=25
step_size=25
temp=0.0

## Run
python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --dataset $dataset \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
    --iter_steps $iter_steps \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --rho $rho \
    --mu $mu \
    --sigma $sigma \
    --eps_bsz $eps_bsz \
    --wandb $wandb \
    --seed $seed \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size \
    --temp $temp \
    --global_verifier $global_verifier \
    --start $start \
    --step_size $step_size \