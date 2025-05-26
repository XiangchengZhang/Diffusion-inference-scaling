from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import ogbench.powderworld  # noqa
from ogbench.powderworld.behaviors import FillBehavior, LineBehavior, SquareBehavior

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'powderworld-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode.')
flags.DEFINE_float('p_random_action', 0.5, 'Probability of selecting a random action.')


def main(_):
    assert FLAGS.dataset_type in ['play']

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
    )
    env.reset()

    # Initialize agents.
    agents = [
        FillBehavior(env=env),
        LineBehavior(env=env),
        SquareBehavior(env=env),
    ]
    probs = np.array([1, 3, 3])  # Agent selection probabilities.
    probs = probs / probs.sum()

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        ob, info = env.reset()
        agent = np.random.choice(agents, p=probs)
        agent.reset(ob, info)

        done = False
        step = 0

        action_step = 0  # Action cycle counter (0, 1, 2).
        while not done:
            if action_step == 0:
                # Select an action every 3 steps.
                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    semantic_action = env.unwrapped.sample_semantic_action()
                else:
                    # Get an action from the agent.
                    semantic_action = agent.select_action(ob, info)
            action = env.unwrapped.semantic_action_to_action(*semantic_action)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if agent.done and FLAGS.dataset_type == 'play':
                agent = np.random.choice(agents, p=probs)
                agent.reset(ob, info)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)

            ob = next_ob
            step += 1
            action_step = (action_step + 1) % 3

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif 'actions':
            dtype = np.int32
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
