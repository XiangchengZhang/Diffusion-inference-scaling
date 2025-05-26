import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ogbench.manipspace import lie


class PlanOracle:
    """Non-Markovian oracle that follows a pre-computed plan.

    It first generates a plan by interpolating the keyframes of the task and adds temporally correlated noise. Then, it
    computes the actions by computing the difference between the current state and the next state in the plan.
    """

    def __init__(self, env, segment_dt=0.4, noise=0.1, noise_smoothing=0.5):
        """Initialize the oracle.

        Args:
            env: Environment.
            segment_dt: Default duration of each segment between keyframes in the plan.
            noise: Noise level to add to the plan.
            noise_smoothing: Noise smoothing level.
        """
        self._env = env
        self._env_dt = self._env.unwrapped._control_timestep
        self._dt = segment_dt
        self._noise = noise
        self._noise_smoothing = noise_smoothing

        self._done = False
        self._t_init = None
        self._t_max = None
        self._plan = None

    def above(self, pose, z):
        return (
            lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3.identity(),
                translation=np.array([0.0, 0.0, z]),
            )
            @ pose
        )

    def to_pose(self, pos, yaw):
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(yaw),
            translation=pos,
        )

    def get_yaw(self, pose):
        yaw = pose.rotation().compute_yaw_radians()
        if yaw < 0.0:
            return yaw + 2 * np.pi
        return yaw

    def shortest_yaw(self, eff_yaw, obj_yaw, translation, n=4):
        """Find the symmetry-aware shortest yaw angle to the object."""
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(symmetries[d]),
            translation=translation,
        )

    def compute_plan(self, times, poses, grasps):
        # Interpolate grasps.
        grasp_interp = interp1d(times, grasps, kind='linear', axis=0, assume_sorted=True)

        # Interpolate poses.
        xyzs = [p.translation() for p in poses]
        xyz_interp = interp1d(times, xyzs, kind='linear', axis=0, assume_sorted=True)

        # Interpolate orientations.
        quats = [p.rotation() for p in poses]

        def quat_interp(t):
            s = np.searchsorted(times, t, side='right') - 1
            interp_time = (t - times[s]) / (times[s + 1] - times[s])
            interp_time = np.clip(interp_time, 0.0, 1.0)
            return lie.interpolate(quats[s], quats[s + 1], interp_time)

        # Generate the plan.
        plan = []
        t = 0.0
        while t < self._t_max:
            action = np.zeros(5)
            action[:3] = xyz_interp(t)
            action[3] = quat_interp(t).compute_yaw_radians()
            action[4] = grasp_interp(t)
            plan.append(action)
            t += self._env_dt

        plan = np.array(plan)

        # Add temporally correlated noise to the plan.
        if self._noise > 0:
            noise = np.random.normal(0, 1, size=(len(plan), 5)) * np.array([0.05, 0.05, 0.05, 0.3, 1.0]) * self._noise
            noise = gaussian_filter1d(noise, axis=0, sigma=self._noise_smoothing)
            plan += noise

        return plan

    @property
    def done(self):
        return self._done

    def reset(self, ob, info):
        pass

    def select_action(self, ob, info):
        # Find the current plan index.
        cur_plan_idx = int((info['time'][0] - self._t_init + 1e-7) // self._env_dt)
        if cur_plan_idx >= len(self._plan) - 1:
            cur_plan_idx = len(self._plan) - 1
            self._done = True

        # Compute the difference between the current state and the current plan.
        ab_action = self._plan[cur_plan_idx]
        action = np.zeros(5)
        action[:3] = ab_action[:3] - info['proprio/effector_pos']
        action[3] = ab_action[3] - info['proprio/effector_yaw'][0]
        action[4] = ab_action[4] - info['proprio/gripper_opening'][0]
        action = self._env.unwrapped.normalize_action(action)

        return action
