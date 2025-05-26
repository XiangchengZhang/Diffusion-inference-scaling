CUDA_VISIBLE_DEVICES=0
data_type=image
image_size=256
dataset="celebahq"
model_name_or_path='google/ddpm-ema-celebahq-256'

task="label_guidance+label_guidance"
guide_network="nateraw/vit-age-classifier+rizvandwiki/gender-classification-2"
target="0+1"

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=4
num_samples=4
logging_resolution=512
guidance_name='tfg'
eval_batch_size=4
wandb=False

rho=0.5
mu=8
sigma=1
eps_bsz=1
iter_steps=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --dataset $dataset \
    --iter_steps $iter_steps \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
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
    --eval_batch_size $eval_batch_size


