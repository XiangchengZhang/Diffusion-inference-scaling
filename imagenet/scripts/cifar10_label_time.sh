CUDA_VISIBLE_DEVICES=0
data_type=image
task=label_guidance_time
image_size=32
dataset="cifar10"
model_name_or_path='openai_cifar10.pt'
guide_network='timeclassifier_cifar10.pt'
train_steps=1000
inference_steps=50
eta=1.0
target=8
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=128
num_samples=128
logging_resolution=512
guidance_name='tfg'
guidance_strength=1.0
eval_batch_size=512
wandb=False


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --guidance_strength $guidance_strength \
    --dataset $dataset \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --wandb $wandb \
    --seed $seed \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size


