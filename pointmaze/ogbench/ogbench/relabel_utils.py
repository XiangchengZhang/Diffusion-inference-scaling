import numpy as np


def relabel_dataset(env_name, env, dataset):
    """Relabel the dataset with rewards and masks based on the fixed task of the environment.
    This is useful for single-task variants of the environments.

    Args:
        env_name: Name of the environment.
        env: Environment.
        dataset: Dataset dictionary.
    """
    assert env.unwrapped._reward_task_id is not None, 'The environment is not in the single-task mode.'
    env.reset()  # Set the task.

    if 'maze' in env_name or 'soccer' in env_name:
        # Locomotion environments.
        qpos_xy_start_idx = 0
        qpos_ball_start_idx = 15
        goal_xy = env.unwrapped.cur_goal_xy
        goal_tol = env.unwrapped._goal_tol

        # Compute successes.
        if 'maze' in env_name:
            dists = np.linalg.norm(dataset['qpos'][:, qpos_xy_start_idx : qpos_xy_start_idx + 2] - goal_xy, axis=-1)
        else:
            dists = np.linalg.norm(dataset['qpos'][:, qpos_ball_start_idx : qpos_ball_start_idx + 2] - goal_xy, axis=-1)
        successes = (dists <= goal_tol).astype(np.float32)

        rewards = successes - 1.0
        masks = 1.0 - successes
    elif 'cube' in env_name or 'scene' in env_name or 'puzzle' in env_name:
        # Manipulation environments.
        qpos_obj_start_idx = 14
        qpos_cube_length = 7

        if 'cube' in env_name:
            num_cubes = env.unwrapped._num_cubes
            target_cube_xyzs = env.unwrapped._data.mocap_pos.copy()

            # Compute successes.
            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    dataset['qpos'][
                        :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
                    ]
                )
            cube_xyzs = np.stack(cube_xyzs_list, axis=1)
            successes = np.linalg.norm(target_cube_xyzs - cube_xyzs, axis=-1) <= 0.04
        elif 'scene' in env_name:
            num_cubes = env.unwrapped._num_cubes
            num_buttons = env.unwrapped._num_buttons
            qpos_drawer_idx = qpos_obj_start_idx + num_cubes * qpos_cube_length + num_buttons
            qpos_window_idx = qpos_drawer_idx + 1
            target_cube_xyzs = env.unwrapped._data.mocap_pos.copy()
            target_button_states = env.unwrapped._target_button_states.copy()
            target_drawer_pos = env.unwrapped._target_drawer_pos
            target_window_pos = env.unwrapped._target_window_pos

            # Compute successes.
            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    dataset['qpos'][
                        :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
                    ]
                )
            cube_xyzs = np.stack(cube_xyzs_list, axis=1)
            cube_successes = np.linalg.norm(target_cube_xyzs - cube_xyzs, axis=-1) <= 0.04
            button_successes = dataset['button_states'] == target_button_states
            drawer_success = np.abs(dataset['qpos'][:, qpos_drawer_idx] - target_drawer_pos) <= 0.04
            window_success = np.abs(dataset['qpos'][:, qpos_window_idx] - target_window_pos) <= 0.04
            successes = np.concatenate(
                [cube_successes, button_successes, drawer_success[:, None], window_success[:, None]], axis=-1
            )
        elif 'puzzle' in env_name:
            num_buttons = env.unwrapped._num_buttons
            target_button_states = env.unwrapped._target_button_states.copy()

            # Compute successes.
            successes = dataset['button_states'] == target_button_states

        rewards = successes.sum(axis=-1) - successes.shape[-1]
        masks = 1.0 - np.all(successes, axis=-1)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    dataset['rewards'] = rewards.astype(np.float32)
    dataset['masks'] = masks.astype(np.float32)


def add_oracle_reps(env_name, env, dataset):
    """Add oracle goal representations to the dataset.

    Args:
        env_name: Name of the environment.
        env: Environment.
        dataset: Dataset dictionary.
    """
    if 'maze' in env_name or 'soccer' in env_name:
        # Locomotion environments.
        qpos_xy_start_idx = 0
        qpos_ball_start_idx = 15

        if 'maze' in env_name:
            oracle_reps = dataset['qpos'][:, qpos_xy_start_idx : qpos_xy_start_idx + 2]
        else:
            oracle_reps = dataset['qpos'][:, qpos_ball_start_idx : qpos_ball_start_idx + 2]
    elif 'cube' in env_name or 'scene' in env_name or 'puzzle' in env_name:
        # Manipulation environments.
        qpos_obj_start_idx = 14
        qpos_cube_length = 7
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        drawer_scaler = 18.0
        window_scaler = 15.0

        if 'cube' in env_name:
            num_cubes = env.unwrapped._num_cubes

            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    dataset['qpos'][
                        :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
                    ]
                )
            cube_xyzs = np.stack(cube_xyzs_list, axis=1)
            oracle_reps = ((cube_xyzs - xyz_center) * xyz_scaler).reshape(-1, num_cubes * 3)
        elif 'scene' in env_name:
            num_cubes = env.unwrapped._num_cubes
            num_buttons = env.unwrapped._num_buttons
            qpos_drawer_idx = qpos_obj_start_idx + num_cubes * qpos_cube_length + num_buttons
            qpos_window_idx = qpos_drawer_idx + 1

            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    dataset['qpos'][
                        :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
                    ]
                )
            cube_xyzs = np.stack(cube_xyzs_list, axis=1)
            cube_reps = ((cube_xyzs - xyz_center) * xyz_scaler).reshape(-1, num_cubes * 3)
            button_reps = dataset['button_states'].copy()
            drawer_reps = dataset['qpos'][:, [qpos_drawer_idx]] * drawer_scaler
            window_reps = dataset['qpos'][:, [qpos_window_idx]] * window_scaler
            oracle_reps = np.concatenate([cube_reps, button_reps, drawer_reps, window_reps], axis=-1)
        elif 'puzzle' in env_name:
            oracle_reps = dataset['button_states'].copy()
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    dataset['oracle_reps'] = oracle_reps.astype(np.float32)
