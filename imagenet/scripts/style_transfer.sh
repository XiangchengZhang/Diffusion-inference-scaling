CUDA_VISIBLE_DEVICES=0
data_type=text2image
image_size=512
dataset="parti_prompts"
model_name_or_path='runwayml/stable-diffusion-v1-5'

task=style_transfer
guide_network='openai/clip-vit-base-patch16'
target=./data/wikiart/1.png

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=False
seed=42
logging_dir='logs'
per_sample_batch_size=1
num_samples=1
logging_resolution=512
guidance_name='tfg'
eval_batch_size=1
wandb=False

rho=0.25
mu=2
sigma=0.1
eps_bsz=1
recur_steps=1
iter_steps=4

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
    --iter_steps $iter_steps \
    --recur_steps $recur_steps \
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


