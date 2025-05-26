import json
import numpy as np
from os.path import join
import os
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from search.configs import Arguments
from search.base_policy import BasePolicy
from search.methods.base_guidance import BaseGuidance


        


class BasePipe:
    def __init__(self, args: Arguments, policy: BasePolicy, guidance: BaseGuidance=None, **kwargs):
        self.args = args
        self.policy = policy
        self.guidance = guidance
        self.setup(**kwargs)

    def setup(self, **kwargs):

        class Parser(utils.Parser):
            dataset: str = self.args.dataset
            config: str = 'config.pointmaze'
        
        #---------------------------------- setup ----------------------------------#

        args = Parser().parse_args('plan')
        args.diffusion_epoch = kwargs.get('diffusion_epoch', args.diffusion_epoch)

        # logger = utils.Logger(args)

        env = datasets.load_environment(args.dataset)

        #---------------------------------- loading ----------------------------------#

        diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch, device=self.args.device)

        diffusion = diffusion_experiment.ema
        diffusion.horizon = self.args.sampling_horizon   ## change the horizon at sampling time
        dataset = diffusion_experiment.dataset
        renderer = diffusion_experiment.renderer

        policy = Policy(diffusion, dataset.normalizer)

        #----------------------------------- register attr -----------------------------#
        self.env = env
        self.env_policy = policy
        self.env_diffusion = diffusion
        self.env_args = args

        self.env_renderer = renderer
        self.env_diffusion_experiment = diffusion_experiment

        #---------------------------------- setup policy -------------------------------#
        self.policy.setup(policy, diffusion, **kwargs)
    
 

    def sample(self, **kwargs):
        env = self.env
        args = self.env_args
        renderer = self.env_renderer

        task_id = kwargs.get('task_id', self.args.task)
        savepath = kwargs.get('savepath', args.savepath)

        #---------------------------------- main loop ----------------------------------#

        observation, info = env.reset(options={"task_id": task_id})
        target = info["goal"]

        cond = {
            self.env_diffusion.horizon - 1: np.array([*target]),
            0: observation
        }

        total_reward = 0

        (action, samples), extras = self.policy(cond, guidance=self.guidance, **kwargs)

        sequence = samples.observations[0]  ## do not support parallel for now


        
        ## observations for rendering
        rollout = [observation.copy()]


        for t in range(env.max_episode_steps):
            next_waypoint = sequence[t+1] if t < len(sequence) - 1 else sequence[-1]
            action = (next_waypoint - observation) * 5.
            action = np.clip(action, -1, 1)
            next_observation, reward, terminal, truncated, _ = env.step(action)
            total_reward += reward
            # score = env.get_normalized_score(total_reward)
            # print(
            #     f't: {t} | r: {reward:.2f} | R: {total_reward:.2f} | {action}'
            # )

            if 'pointmaze' in args.dataset:
                xy = next_observation[:2]
                goal = env.cur_task_info['goal_xy']
                # print(
                #     f'maze | pos: {xy} | goal: {goal}'
                # )

            ## update rollout observations
            rollout.append(next_observation.copy())

            # logger.log(score=score, step=t)
            if t == 0: 
                fullpath = join(savepath, f'{t}.png')
                renderer.composite(fullpath, samples.observations, ncol=1)
                np.savez(join(savepath, 'plan.npz'), plan=samples.observations)

            # if t % args.vis_freq == 0 or terminal or truncated:
           
                # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                ## save rollout thus far
                # renderer.composite(join(savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
                # np.savez(join(savepath,'rollout.npz'), rollout=np.array(rollout)[None])
                # input()
                # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

                # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

            if terminal or truncated:
                break
                

            observation = next_observation
        
        ## to save computation, only save the last frame
        renderer.composite(join(savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
        np.savez(join(savepath,'rollout.npz'), rollout=np.array(rollout)[None])

        ## save result as a json file
        json_path = join(savepath, 'rollout.json')
        json_data = {'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': self.env_diffusion_experiment.epoch, }
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
        extras['total_reward'] = total_reward

        return extras


    def eval(self, num_samples, **kwargs):
        total_results = {}
        path = kwargs.pop('path', self.args.logging_dir)
        
        for i in range(num_samples):
            counter = len(os.listdir(path)) - 1
            savepath = join(path, f'{counter}')
            os.makedirs(savepath, exist_ok=True)
            kwargs['savepath'] = savepath
            results = self.sample(**kwargs)
            for key, value in results.items():
                if key not in total_results:
                    total_results[key] = 0
                total_results[key] += value
        
        for key in total_results:
            total_results[key] /= num_samples
            if total_results[key] <= 1: total_results[key] *= 100
        
        return total_results
        
    
    def experiment(self, **kwargs):
        import time
        exp_name = self.env_args.exp_name
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        path = f"{self.args.logging_dir}/{self.args.dataset}/inference/{exp_name}/{cur_time}/"
        os.makedirs(path, exist_ok=True)
        with open(join(path, 'args.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=2, sort_keys=True)
        super(utils.Parser, self.env_args).save(f"{path}/env_args.json",skip_unpicklable=True)

        returns = {}
        tasks = self.args.task
        for task_id in tasks:  
            total_results = self.eval(self.args.num_samples, path=path, task_id=task_id)
            returns[task_id] = total_results
            result_str = f"Task: {task_id} | Success Rate: {total_results['total_reward']:.2f}"
            with open(join(path, 'results.txt'), 'a') as f:
                f.write(result_str + '\n')
        returns['average'] = {}
        for key in total_results:
            returns['average'][key] = np.mean([returns[task][key] for task in tasks]).item()
           
        return returns

        


        


