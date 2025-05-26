import numpy as np


class MarkovOracle:
    """Markovian oracle for manipulation tasks."""

    def __init__(self, env, min_norm=0.4):
        """Initialize the oracle.

        Args:
            env: Environment.
            min_norm: Minimum norm for the relative position. Setting it to a non-zero value can help the agent to learn
                more robust policies.
        """
        self._env = env
        self._min_norm = min_norm
        self._debug = False  # Set to True to print debug information.
        self._done = False

        if self._debug:
            np.set_printoptions(suppress=True)

    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm

    def shortest_yaw(self, eff_yaw, obj_yaw, n=4):
        """Find the symmetry-aware shortest yaw angle to the object."""
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return symmetries[d]

    def print_phase(self, phase):
        """Print the current phase."""
        if self._debug:
            print(f'Phase {phase:50}', end=' ')

    @property
    def done(self):
        return self._done

    def reset(self, ob, info):
        pass

    def select_action(self, ob, info):
        pass
