import mujoco
import numpy as np

PI = np.pi
PI_2 = 2 * np.pi


def angle_diff(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return np.mod(q1 - q2 + PI, PI_2) - PI


class DiffIKController:
    """Differential inverse kinematics controller."""

    def __init__(
        self,
        model: mujoco.MjModel,
        sites: list,
        qpos0: np.ndarray = None,
        damping_coeff: float = 1e-12,
        max_angle_change: float = np.radians(45),
    ):
        self._model = model
        self._data = mujoco.MjData(self._model)
        self._qp0 = qpos0
        self._max_angle_change = max_angle_change

        # Cache references.
        self._ns = len(sites)  # Number of sites.
        self._site_ids = np.asarray([self._model.site(s).id for s in sites])

        # Preallocate arrays.
        self._err = np.empty((self._ns, 6))
        self._site_quat = np.empty((self._ns, 4))
        self._site_quat_inv = np.empty((self._ns, 4))
        self._err_quat = np.empty((self._ns, 4))
        self._jac = np.empty((6 * self._ns, self._model.nv))
        self._damping = damping_coeff * np.eye(6 * self._ns)
        self._eye = np.eye(self._model.nv)

    def _forward_kinematics(self) -> None:
        """Minimal computation required for forward kinematics."""
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)  # Required for mj_jacSite.

    def _integrate(self, update: np.ndarray) -> None:
        """Integrate the joint velocities in-place."""
        mujoco.mj_integratePos(self._model, self._data.qpos, update, 1.0)

    def _compute_translational_error(self, pos: np.ndarray) -> None:
        """Compute the error between the desired and current site positions."""
        self._err[:, :3] = pos - self._data.site_xpos[self._site_ids]

    def _compute_rotational_error(self, quat: np.ndarray) -> None:
        """Compute the error between the desired and current site orientations."""
        for i, site_id in enumerate(self._site_ids):
            mujoco.mju_mat2Quat(self._site_quat[i], self._data.site_xmat[site_id])
            mujoco.mju_negQuat(self._site_quat_inv[i], self._site_quat[i])
            mujoco.mju_mulQuat(self._err_quat[i], quat[i], self._site_quat_inv[i])
            mujoco.mju_quat2Vel(self._err[i, 3:], self._err_quat[i], 1.0)

    def _compute_jacobian(self) -> None:
        """Update site end-effector Jacobians."""
        for i, site_id in enumerate(self._site_ids):
            jacp = self._jac[6 * i : 6 * i + 3]
            jacr = self._jac[6 * i + 3 : 6 * i + 6]
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)

    def _error_threshold_reached(self, pos_thresh: float, ori_thresh: float) -> bool:
        """Return True if position and rotation errors are below the thresholds."""
        pos_achieved = np.linalg.norm(self._err[:, :3]) <= pos_thresh
        ori_achieved = np.linalg.norm(self._err[:, 3:]) <= ori_thresh
        return pos_achieved and ori_achieved

    def _solve(self) -> np.ndarray:
        """Solve for joint velocities using damped least squares."""
        H = self._jac @ self._jac.T + self._damping
        x = self._jac.T @ np.linalg.solve(H, self._err.ravel())
        if self._qp0 is not None:
            jac_pinv = np.linalg.pinv(H)
            q_err = angle_diff(self._qp0, self._data.qpos)
            x += (self._eye - (self._jac.T @ jac_pinv) @ self._jac) @ q_err
        return x

    def _scale_update(self, update: np.ndarray) -> np.ndarray:
        """Scale down update so that the max allowable angle change is not exceeded."""
        update_max = np.max(np.abs(update))
        if update_max > self._max_angle_change:
            update *= self._max_angle_change / update_max
        return update

    def solve(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        curr_qpos: np.ndarray,
        max_iters: int = 20,
        pos_thresh: float = 1e-4,
        ori_thresh: float = 1e-4,
    ) -> np.ndarray:
        self._data.qpos = curr_qpos

        for _ in range(max_iters):
            self._forward_kinematics()

            self._compute_translational_error(np.atleast_2d(pos))
            self._compute_rotational_error(np.atleast_2d(quat))
            if self._error_threshold_reached(pos_thresh, ori_thresh):
                break

            self._compute_jacobian()
            update = self._scale_update(self._solve())
            self._integrate(update)

        return self._data.qpos.copy()
