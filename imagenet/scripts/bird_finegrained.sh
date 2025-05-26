CUDA_VISIBLE_DEVICES=0
data_type=image
image_size=256
dataset="bird-species"
model_name_or_path='models/openai_imagenet.pt'

task=label_guidance
guide_network='dennisjooo/Birds-Classifier-EfficientNetB2'
target=111

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=2
num_samples=2
logging_resolution=512
guidance_name='tfg'
eval_batch_size=2
wandb=False

rho=0.5
mu=0.5
sigma=0.01
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


