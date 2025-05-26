from gymnasium.envs.registration import register

register(
    id='powderworld-easy-v0',
    entry_point='ogbench.powderworld.powderworld_env:PowderworldEnv',
    max_episode_steps=500,
    kwargs=dict(num_elems=2),
)

register(
    id='powderworld-medium-v0',
    entry_point='ogbench.powderworld.powderworld_env:PowderworldEnv',
    max_episode_steps=500,
    kwargs=dict(num_elems=5),
)

register(
    id='powderworld-hard-v0',
    entry_point='ogbench.powderworld.powderworld_env:PowderworldEnv',
    max_episode_steps=500,
    kwargs=dict(num_elems=8),
)
