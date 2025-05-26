


## D4RL experiments

The code base is adapted from [CEP-energy-guided-diffusion](https://github.com/thu-ml/CEP-energy-guided-diffusion#)

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed.

### Running
To pretrain the behavior model, run

```.bash
$ TASK="walker2d-medium-expert-v2"; seed=0; setting="reproduce"; python3 -u train_behavior.py --expid $TASK${seed}${setting} --env $TASK --seed ${seed}
```

The pretrained behavior model will be stored in the `./models_rl/`. Once we have the pretrained checkpoint at `/path/to/pretrained/ckpt.pth` ([download url](https://drive.google.com/drive/folders/1snFcmcJaalcCWW9roBjeCjpWjpCeDM_P?usp=drive_link)), we can train the critic model:

```.bash
$ TASK="walker2d-medium-expert-v2"; seed=0; setting="reproduce"; python3 -u train_critic.py --actor_load_path /path/to/pretrained/ckpt.pth --expid $TASK${seed}${setting} --env $TASK --diffusion_steps 15 --seed ${seed} --alpha 3 --q_alpha 1 --method "CEP"
```

### Inference
To evaluate test-time search in our method, run `exp_tts.py`. To evaluate QGPO with time dependent energy guidance, run `exp_cep.py`