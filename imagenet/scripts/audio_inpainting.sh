CUDA_VISIBLE_DEVICES=0
data_type=audio
image_size=256
dataset="teticio/audio-diffusion-256"
model_name_or_path='teticio/audio-diffusion-256'

task=audio_inpainting
guide_network='no'
target=no

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='audiologs'
per_sample_batch_size=4
num_samples=4
logging_resolution=512
guidance_name='tfg'
eval_batch_size=16
wandb=False

rho=0.25
mu=2
sigma=0.1
eps_bsz=1
guidance_strength=8.0
iter_steps=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --iter_steps $iter_steps \
    --dataset $dataset \
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
    --guidance_strength $guidance_strength \
    --eval_batch_size $eval_batch_size


