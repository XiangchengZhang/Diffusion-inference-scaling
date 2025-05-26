CUDA_VISIBLE_DEVICES=1
data_type=image
image_size=256
dataset="cat"
model_name_or_path='google/ddpm-ema-cat-256'

task=super_resolution
guide_network='no'
target=no

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=16
num_samples=256
logging_resolution=512
guidance_name='tfg'
eval_batch_size=16
wandb=False
recur_steps=1
iter_steps=4

rho=4
mu=2
sigma=0.01
eps_bsz=1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
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
    --eta $eta \
    --iter_steps $iter_steps \
    --recur_steps $recur_steps \
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
    --eval_batch_size $eval_batch_size


