import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class ButtonPlanOracle(PlanOracle):
    def __init__(self, gripper_always_closed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gripper_always_closed = gripper_always_closed

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        poses['initial'] = plan_input['effector_initial']
        poses['press_start'] = self.above(plan_input['button'], 0.06)
        poses['press'] = self.above(plan_input['button'], -0.025)
        poses['press_end'] = poses['press_start']
        poses['final'] = plan_input['effector_goal']

        # Times.
        times = {}
        distance = np.linalg.norm(poses['initial'].translation() - poses['press_start'].translation())
        times['initial'] = 0.0
        times['press_start'] = times['initial'] + self._dt * (0.5 + distance * 4)
        times['press'] = times['press_start'] + self._dt * 0.8
        times['press_end'] = times['press'] + self._dt * 0.8
        times['final'] = times['press_end'] + self._dt * 1.25
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.1

        # Grasps.
        grasps = {}
        if self._gripper_always_closed:
            g = 1.0
        else:
            g = 0.0
        for name in times.keys():
            if not self._gripper_always_closed:
                if name in {'press_start', 'final'}:
                    g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps

    def reset(self, ob, info):
        plan_input = {
            'effector_initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'effector_goal': self.to_pose(
                pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
                yaw=np.random.uniform(-np.pi, np.pi),
            ),
            'button': self.to_pose(
                pos=info['privileged/target_button_top_pos'],
                yaw=info['privileged/target_button_top_pos'][0],
            ),
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_init = info['time'][0]
        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)
