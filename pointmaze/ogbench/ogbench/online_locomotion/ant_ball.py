import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium.spaces import Box

from ogbench.online_locomotion.ant import AntEnv


class AntBallEnv(AntEnv):
    """Gymnasium Ant environment with a ball."""

    def __init__(self, xml_file=None, *args, **kwargs):
        if xml_file is None:
            xml_file = self.xml_file

        # Add a ball to the environment.
        tree = ET.parse(xml_file)
        worldbody = tree.find('.//worldbody')
        ET.SubElement(
            worldbody,
            'geom',
            name='target',
            type='cylinder',
            size='.4 .05',
            pos='0 0 .05',
            material='target',
            contype='0',
            conaffinity='0',
        )
        ball = ET.SubElement(worldbody, 'body', name='ball', pos='0 0 3')
        ET.SubElement(ball, 'freejoint', name='ball_root')
        ET.SubElement(ball, 'geom', name='ball', size='.25', material='ball', priority='1', conaffinity='1', condim='6')
        ET.SubElement(ball, 'light', name='ball_light', pos='0 0 4', mode='trackcom')

        # Rename the track camera to avoid automatic tracking.
        track_camera = tree.find('.//camera[@name="track"]')
        track_camera.set('name', 'back')
        _, xml_file = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(xml_file)

        super().__init__(xml_file=xml_file, *args, **kwargs)

        self.cur_goal_xy = np.zeros(2)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self._get_obs().shape[0],), dtype=np.float64)

        self.reset()
        self.render()

        # Adjust the camera.
        self.mujoco_renderer.viewer.cam.lookat[0] = 0
        self.mujoco_renderer.viewer.cam.lookat[1] = 0
        self.mujoco_renderer.viewer.cam.distance = 30
        self.mujoco_renderer.viewer.cam.elevation = -90

    def reset(self, options=None, *args, **kwargs):
        ob, info = super().reset(*args, **kwargs)

        agent_init_xy = np.random.uniform(low=-1, high=1, size=2)
        ball_init_xy = np.random.uniform(low=-2, high=2, size=2)
        goal_xy = np.random.uniform(low=-12, high=12, size=2)

        self.set_agent_ball_xy(agent_init_xy, ball_init_xy)
        self.set_goal(goal_xy=goal_xy)
        ob = self._get_obs()

        return ob, info

    def step(self, action):
        prev_agent_xy, prev_ball_xy = self.get_agent_ball_xy()
        goal_xy = self.cur_goal_xy
        prev_agent_ball_dist = np.linalg.norm(prev_agent_xy - prev_ball_xy)
        prev_ball_goal_dist = np.linalg.norm(prev_ball_xy - goal_xy)

        ob, reward, terminated, truncated, info = super().step(action)

        if np.linalg.norm(self.get_agent_ball_xy()[1] - self.cur_goal_xy) <= 0.5:
            info['success'] = 1.0
        else:
            info['success'] = 0.0

        # Compute the distance between the agent and the ball, and the ball and the goal.
        agent_xy, ball_xy = self.get_agent_ball_xy()
        agent_ball_dist = np.linalg.norm(agent_xy - ball_xy)
        ball_goal_dist = np.linalg.norm(ball_xy - goal_xy)

        # Use the change in distances as the reward.
        reward = ((prev_ball_goal_dist - ball_goal_dist) * 2.5 + (prev_agent_ball_dist - agent_ball_dist)) * 10

        return ob, reward, terminated, truncated, info

    def set_goal(self, goal_xy):
        self.cur_goal_xy = goal_xy
        self.model.geom('target').pos[:2] = goal_xy

    def get_agent_ball_xy(self):
        agent_xy = self.data.qpos[:2].copy()
        ball_xy = self.data.qpos[-7:-5].copy()

        return agent_xy, ball_xy

    def set_agent_ball_xy(self, agent_xy, ball_xy):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[:2] = agent_xy
        qpos[-7:-5] = ball_xy
        self.set_state(qpos, qvel)

    def _get_obs(self):
        # Return the agent's position, velocity, the ball's relative position, and the goal's relative position.
        agent_xy, ball_xy = self.get_agent_ball_xy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        return np.concatenate([qpos[2:-7], qpos[-5:], qvel, ball_xy - agent_xy, np.array(self.cur_goal_xy) - ball_xy])
