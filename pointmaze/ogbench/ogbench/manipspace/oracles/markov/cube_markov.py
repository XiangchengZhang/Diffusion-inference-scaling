import numpy as np

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle


class CubeMarkovOracle(MarkovOracle):
    def __init__(self, max_step=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_step = max_step

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._max_step = 200
        self._final_pos = np.random.uniform(*self._env.unwrapped._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)

    def select_action(self, ob, info):
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        gripper_opening = info['proprio/gripper_opening']

        target_block = info['privileged/target_block']
        block_pos = info[f'privileged/block_{target_block}_pos']
        block_yaw = self.shortest_yaw(effector_yaw, info[f'privileged/block_{target_block}_yaw'][0])
        target_pos = info['privileged/target_block_pos']
        target_yaw = self.shortest_yaw(effector_yaw, info['privileged/target_block_yaw'][0])

        block_above_offset = np.array([0, 0, 0.18])
        above_threshold = 0.16
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        gripper_open = info['proprio/gripper_contact'] < 0.1
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.02
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.04
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not target_pos_aligned:
            if not xy_aligned:
                self.print_phase('1: Move above the block')
                action = np.zeros(5)
                diff = block_pos + block_above_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                self.print_phase('2: Move to the block')
                diff = block_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif pos_aligned and not gripper_closed:
                self.print_phase('3: Grasp')
                diff = block_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not above and not target_xy_aligned:
                self.print_phase('4: Move in the air')
                diff = np.array([block_pos[0], block_pos[1], block_above_offset[2] * 2]) - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and above and not target_xy_aligned:
                self.print_phase('5: Move above the target')
                diff = target_pos + block_above_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            else:
                self.print_phase('6: Move to the target')
                diff = target_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
        else:
            if not gripper_open:
                self.print_phase('7: Release')
                diff = target_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = -1
            elif gripper_open and not above:
                self.print_phase('8: Move in the air')
                diff = np.array([block_pos[0], block_pos[1], above_threshold * 2]) - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            else:
                self.print_phase('9: Move to the final position')
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
