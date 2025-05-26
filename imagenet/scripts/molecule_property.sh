#!/bin/bash

target=mu   # such as mu
CUDA_VISIBLE_DEVICES=0
rho_schedule=increase
mu_schedule=decrease
data_type=molecule
task=molecule_property
dataset="qm9"

base_dir=./models/pretrained_models

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=False
seed=42
per_sample_batch_size=32
eval_batch_size=64
wandb=False
wandb_project='qm9_guidance_beam'

# These are just placeholders to be re-written by the sweep
# wandb_project='trails'
guidance_strength=0.1
rho=0.001
mu=0.002
sigma=0.1
eps_bsz=1
# These are fixed hyperparameters for large scale running
num_samples=32
logging_dir='logs'

# Fixed hyperparameters during the sweep
sigma_schedule='decrease'

sweep_dir='sweep_qm9_guidance_again'
cuda_ids=$CUDA_VISIBLE_DEVICES

generators_path=$base_dir/EDMsecond/generative_model_ema.npy
args_generators_path=$base_dir/EDMsecond/args.pickle

energy_path=$base_dir/tf_predict_$target/model_ema_2000.npy
args_energy_path=$base_dir/tf_predict_$target/args_2000.pickle

classifiers_path=$base_dir/evaluate_$target/best_checkpoint.npy
args_classifiers_path=$base_dir/evaluate_$target/args.pickle

# recur_steps=2
iter_steps=4
# for guidance_name in 'dynamic_0.01' 'dynamic_0.05';
guidance_name='tfg'
for recur_steps in 1;
do
    cmd="CUDA_VISIBLE_DEVICES=$cuda_ids python main.py \
        --generators_path $generators_path \
        --args_generators_path $args_generators_path \
        --energy_path $energy_path \
        --task $task \
        --eps_bsz $eps_bsz \
        --args_energy_path $args_energy_path \
        --classifiers_path $classifiers_path \
        --args_classifiers_path $args_classifiers_path \
        --data_type $data_type \
        --wandb $wandb \
        --iter_steps $iter_steps \
        --wandb_project $wandb_project \
        --dataset $dataset \
        --train_steps $train_steps \
        --inference_steps $inference_steps \
        --target $target \
        --eta $eta \
        --clip_x0 $clip_x0 \
        --guidance_strength $guidance_strength \
        --recur_steps $recur_steps \
        --seed $seed \
        --logging_dir $logging_dir \
        --per_sample_batch_size $per_sample_batch_size \
        --num_samples $num_samples \
        --guidance_name $guidance_name \
        --eval_batch_size $eval_batch_size \
        --rho $rho \
        --mu $mu \
        --sigma $sigma \
        --rho_schedule $rho_schedule \
        --mu_schedule $mu_schedule \
        --sigma_schedule $sigma_schedule
    "
    echo $cmd
    eval "$cmd"
    wait
done
