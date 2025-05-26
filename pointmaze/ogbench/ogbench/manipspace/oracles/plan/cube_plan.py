import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class CubePlanOracle(PlanOracle):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}

        # Pick.
        block_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['block_initial']),
            translation=plan_input['block_initial'].translation(),
        )
        poses['initial'] = plan_input['effector_initial']
        poses['pick'] = self.above(block_initial, 0.1 + np.random.uniform(0, 0.1))
        poses['pick_start'] = block_initial
        poses['pick_end'] = block_initial
        poses['postpick'] = poses['pick']

        # Place.
        block_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(poses['postpick']),
            obj_yaw=self.get_yaw(plan_input['block_goal']),
            translation=plan_input['block_goal'].translation(),
        )
        poses['place'] = self.above(block_goal, 0.1 + np.random.uniform(0, 0.1))
        poses['place_start'] = block_goal
        poses['place_end'] = block_goal
        poses['postplace'] = poses['place']
        poses['final'] = plan_input['effector_goal']

        # Clearance.
        midway = lie.interpolate(poses['postpick'], poses['place'])
        poses['clearance'] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array([*midway.translation()[:2], poses['initial'].translation()[-1]])
            + np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0.2]),
        )

        # Times.
        times = {}
        times['initial'] = 0.0
        times['pick'] = times['initial'] + self._dt
        times['pick_start'] = times['pick'] + self._dt * 1.5
        times['pick_end'] = times['pick_start'] + self._dt
        times['postpick'] = times['pick_end'] + self._dt
        times['clearance'] = times['postpick'] + self._dt
        times['place'] = times['clearance'] + self._dt
        times['place_start'] = times['place'] + self._dt * 1.5
        times['place_end'] = times['place_start'] + self._dt
        times['postplace'] = times['place_end'] + self._dt
        times['final'] = times['postplace'] + self._dt
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.2

        # Grasps.
        g = 0.0
        grasps = {}
        for name in times.keys():
            if name in {'pick_end', 'place_end'}:
                g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps

    def reset(self, ob, info):
        target_block = info['privileged/target_block']
        plan_input = {
            'effector_initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'effector_goal': self.to_pose(
                pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
                yaw=np.random.uniform(-np.pi, np.pi),
            ),
            'block_initial': self.to_pose(
                pos=info[f'privileged/block_{target_block}_pos'],
                yaw=info[f'privileged/block_{target_block}_yaw'][0],
            ),
            'block_goal': self.to_pose(
                pos=info['privileged/target_block_pos'],
                yaw=info['privileged/target_block_yaw'][0],
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
