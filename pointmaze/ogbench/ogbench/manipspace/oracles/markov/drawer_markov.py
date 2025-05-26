import numpy as np

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle


class DrawerMarkovOracle(MarkovOracle):
    def __init__(self, max_step=75, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_step = max_step

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._final_pos = np.random.uniform(*self._env.unwrapped._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)

    def select_action(self, ob, info):
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        gripper_opening = info['proprio/gripper_opening']

        drawer_pos = info['privileged/drawer_handle_pos']
        drawer_yaw = self.shortest_yaw(effector_yaw, info['privileged/drawer_handle_yaw'][0], n=2)
        target_pos = info['privileged/target_drawer_handle_pos']

        drawer_above_offset = np.array([0, 0, 0.12])
        above_threshold = 0.18
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(drawer_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(drawer_pos - effector_pos) <= 0.03
        target_pos_aligned = np.linalg.norm(target_pos - drawer_pos) <= 0.01
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not target_pos_aligned:
            if not xy_aligned:
                self.print_phase('1: Move above the drawer handle')
                action = np.zeros(5)
                diff = drawer_pos + drawer_above_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (drawer_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                self.print_phase('2: Move to the drawer handle')
                diff = drawer_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (drawer_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            else:
                self.print_phase('3: Move to the target')
                diff = target_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (drawer_yaw - effector_yaw) * gain_yaw
                action[4] = -1
        else:
            if not above:
                self.print_phase('4: Move in the air')
                diff = (
                    np.array(
                        [
                            drawer_pos[0],
                            drawer_pos[1],
                            above_threshold * 2,
                        ]
                    )
                    - effector_pos
                )
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            else:
                self.print_phase('5: Move to the final position')
                diff = self._final_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1

            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        if self._debug:
            print(action)

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action
