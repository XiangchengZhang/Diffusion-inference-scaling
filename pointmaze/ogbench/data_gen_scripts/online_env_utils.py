import gymnasium
from utils.env_utils import EpisodeMonitor


def make_online_env(env_name):
    """Make online environment.

    If the environment name contains the '-xy' suffix, the environment will be wrapped with a directional locomotion
    wrapper. For example, 'online-ant-xy-v0' will return an 'online-ant-v0' environment wrapped with GymXYWrapper.

    Args:
        env_name: Name of the environment.
    """
    import ogbench.online_locomotion  # noqa

    # Manually recognize the '-xy' suffix, which indicates that the environment should be wrapped with a directional
    # locomotion wrapper.
    if '-xy' in env_name:
        env_name = env_name.replace('-xy', '')
        apply_xy_wrapper = True
    else:
        apply_xy_wrapper = False

    # Set camera.
    if 'humanoid' in env_name:
        extra_kwargs = dict(camera_id=0)
    else:
        extra_kwargs = dict()

    # Make environment.
    env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200, **extra_kwargs)

    if apply_xy_wrapper:
        # Apply the directional locomotion wrapper.
        from ogbench.online_locomotion.wrappers import DMCHumanoidXYWrapper, GymXYWrapper

        if 'humanoid' in env_name:
            env = DMCHumanoidXYWrapper(env, resample_interval=200)
        else:
            env = GymXYWrapper(env, resample_interval=100)

    env = EpisodeMonitor(env)

    return env
