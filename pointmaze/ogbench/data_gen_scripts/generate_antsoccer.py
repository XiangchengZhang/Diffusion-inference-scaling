import glob
import json
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from agents import SACAgent
from tqdm import trange
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antsoccer-arena-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'navigate', 'Dataset type.')
flags.DEFINE_string('loco_restore_path', 'experts/ant', 'Locomotion agent restore path.')
flags.DEFINE_integer('loco_restore_epoch', 400000, 'Locomotion agent restore epoch.')
flags.DEFINE_string('ball_restore_path', 'experts/antball', 'Ball agent restore path.')
flags.DEFINE_integer('ball_restore_epoch', 12000000, 'Ball agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode.')


def load_agent(restore_path, restore_epoch, ob_dim, action_dim):
    """Initialize and load a SAC agent from a given path."""
    # Load agent config.
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    with open(candidates[0] + '/flags.json', 'r') as f:
        agent_config = json.load(f)['agent']

    # Load agent.
    agent = SACAgent.create(
        FLAGS.seed,
        np.zeros(ob_dim),
        np.zeros(action_dim),
        agent_config,
    )
    agent = restore_agent(agent, restore_path, restore_epoch)

    return agent


def main(_):
    assert FLAGS.dataset_type in ['navigate', 'stitch']
    # 'navigate': Repeatedly navigate to the ball and then to a goal in a single episode.
    # 'stitch': Either only navigate or only dribble the ball to a goal in a single episode.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    ob_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize oracle agent.
    loco_agent = load_agent(FLAGS.loco_restore_path, FLAGS.loco_restore_epoch, ob_dim, action_dim)
    ball_agent = load_agent(FLAGS.ball_restore_path, FLAGS.ball_restore_epoch, ob_dim, action_dim)
    loco_actor_fn = supply_rng(loco_agent.sample_actions, rng=loco_agent.rng)
    ball_actor_fn = supply_rng(ball_agent.sample_actions, rng=ball_agent.rng)

    def get_agent_action(ob, goal_xy):
        """Get an action for the agent to navigate to the goal."""
        if 'arena' not in FLAGS.env_name:
            # In the actual maze environment, replace the goal with the oracle subgoal.
            goal_xy, _ = env.unwrapped.get_oracle_subgoal(ob[:2], goal_xy)
        goal_dir = goal_xy - ob[:2]
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
        # Concatenate the agent's joint positions (excluding the x-y position), joint velocities, and goal direction.
        agent_ob = np.concatenate([ob[2:15], ob[22:36], goal_dir])
        action = loco_actor_fn(agent_ob, temperature=0)
        return action

    def get_ball_action(ob, ball_xy, goal_xy):
        """Get an action for the agent to dribble the ball to the goal."""
        if 'arena' in FLAGS.env_name:
            if np.linalg.norm(goal_xy - ball_xy) > 10:
                # If the ball is too far from the goal, set a virtual goal 10 units away from the ball. This is because
                # the ball agent is not trained to dribble the ball to the goal that is too far away.
                goal_xy = ball_xy + 10 * (goal_xy - ball_xy) / np.linalg.norm(goal_xy - ball_xy)
        else:
            # In the actual maze environment, replace the goal with the oracle subgoal.
            goal_xy, _ = env.unwrapped.get_oracle_subgoal(ball_xy, goal_xy)
        # Concatenate the agent and ball's joint positions (excluding their x-y positions), their joint velocities, and
        # the relative positions of the ball and the goal.
        agent_ob = np.concatenate([ob[2:15], ob[17:], ball_xy - agent_xy, goal_xy - ball_xy])
        action = ball_actor_fn(agent_ob, temperature=0)
        return action

    # Store all empty cells.
    all_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        if FLAGS.dataset_type == 'navigate':
            # Sample random initial positions for the agent, the ball, and the goal.
            agent_init_idx, ball_init_idx, goal_idx = np.random.choice(len(all_cells), 3, replace=False)
            agent_init_ij = all_cells[agent_init_idx]
            ball_init_ij = all_cells[ball_init_idx]
            goal_ij = all_cells[goal_idx]
        elif FLAGS.dataset_type == 'stitch':
            # Randomly choose between the 'navigate' and 'dribble' modes.
            cur_mode = 'navigate' if np.random.randint(2) == 0 else 'dribble'

            # Sample random initial positions for the agent, the ball, and the goal. In the 'dribble' mode, the ball
            # always starts at the agent's position.
            agent_init_idx, ball_init_idx, goal_idx = np.random.choice(len(all_cells), 3, replace=False)
            agent_init_ij = all_cells[agent_init_idx]
            ball_init_ij = all_cells[ball_init_idx] if cur_mode == 'navigate' else agent_init_ij
            goal_ij = all_cells[goal_idx]
        else:
            raise ValueError(f'Unsupported dataset_type: {FLAGS.dataset_type}')

        ob, _ = env.reset(
            options=dict(task_info=dict(agent_init_ij=agent_init_ij, ball_init_ij=ball_init_ij, goal_ij=goal_ij))
        )

        done = False
        step = 0

        virtual_agent_goal_xy = None  # Virtual goal for the agent to move to when stuck.

        while not done:
            agent_xy, ball_xy = env.unwrapped.get_agent_ball_xy()
            agent_xy, ball_xy = np.array(agent_xy), np.array(ball_xy)
            goal_xy = np.array(env.unwrapped.cur_goal_xy)

            if FLAGS.dataset_type == 'navigate':
                if virtual_agent_goal_xy is None:
                    if np.linalg.norm(agent_xy - ball_xy) > 2:
                        # If the agent is far from the ball, move to the ball.
                        action = get_agent_action(ob, ball_xy)
                    else:
                        # If the agent is close to the ball, dribble the ball to the goal.
                        action = get_ball_action(ob, ball_xy, goal_xy)
                else:
                    # When virtual_agent_goal_xy is set, move to the virtual goal.
                    action = get_agent_action(ob, virtual_agent_goal_xy)
            elif FLAGS.dataset_type == 'stitch':
                if cur_mode == 'navigate':
                    # Navigate to the goal.
                    action = get_agent_action(ob, goal_xy)
                else:
                    # Dribble the ball to the goal.
                    action = get_ball_action(ob, ball_xy, goal_xy)

            # Add Gaussian noise to the action.
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']

            if virtual_agent_goal_xy is not None and np.linalg.norm(virtual_agent_goal_xy - next_ob[:2]) <= 0.5:
                # If the agent reaches the virtual goal, clear it.
                virtual_agent_goal_xy = None

            if FLAGS.dataset_type == 'navigate':
                if success:
                    # Sample a new goal state when the current goal is reached.
                    goal_ij = all_cells[np.random.randint(len(all_cells))]
                    env.unwrapped.set_goal(goal_ij)

                # Determine whether the agent is stuck.
                if (
                    step > 150
                    and virtual_agent_goal_xy is None
                    and np.linalg.norm(np.array(dataset['observations'][-150:])[:, :2] - next_ob[:2], axis=1).max() <= 2
                ):
                    # When the agent is stuck for 150 steps, set a virtual goal to move to a random cell.
                    virtual_agent_goal_ij = all_cells[np.random.randint(len(all_cells))]
                    virtual_agent_goal_xy = np.array(env.unwrapped.ij_to_xy(virtual_agent_goal_ij))

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {
        k: np.array(v[:total_train_steps], dtype=np.float32 if k != 'terminals' else bool) for k, v in dataset.items()
    }
    val_dataset = {
        k: np.array(v[total_train_steps:], dtype=np.float32 if k != 'terminals' else bool) for k, v in dataset.items()
    }

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
