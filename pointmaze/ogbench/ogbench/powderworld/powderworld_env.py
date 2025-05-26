import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from PIL import Image

from ogbench.powderworld.sim import PWRenderer, PWSim, interp, pw_element_names


class PowderworldEnv(gymnasium.Env):
    """Powderworld enviroment.

    This wraps the numpy version of the Powderworld simulator to create a gymnasium environment. It supports
    goal-based tasks where the agent must recreate a given world state.
    """

    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 15,
    }

    def __init__(
        self,
        world_size=32,
        grid_size=4,
        brush_size=4,
        num_elems=5,
        mode='task',
        render_mode=None,
        width=192,
        height=192,
    ):
        """Initialize the Powderworld environment.

        Args:
            world_size: Size of the world grid.
            grid_size: Size of the action grid.
            brush_size: Size of the brush.
            num_elems: Number of elements in the world. Must be 2, 5, or 8.
            mode: Mode of the environment. Either 'task' or 'data_collection'. In 'task' mode, the environment is used
                for training and evaluation. In 'data_collection' mode, the environment is used for collecting offline
                data.
            render_mode: Rendering mode. Unused; for compatibility with `gymnasium`.
            width: Width of the rendered image. Only used for rendering, not for observations.
            height: Height of the rendered image. Only used for rendering, not for observations.
        """
        self.pw = PWSim()
        self.pwr = PWRenderer()

        self._world_size = world_size
        self._grid_size = grid_size
        self._brush_size = brush_size
        self._mode = mode
        self._render_width = width
        self._render_height = height
        self._num_elems = num_elems
        if num_elems == 2:
            self._elem_names = ['plant', 'stone']
        elif num_elems == 5:
            self._elem_names = ['sand', 'water', 'fire', 'plant', 'stone']
        elif num_elems == 8:
            self._elem_names = ['sand', 'water', 'fire', 'plant', 'stone', 'gas', 'wood', 'ice']
        else:
            raise NotImplementedError
        self._elems = [pw_element_names.index(elem_name) for elem_name in self._elem_names]
        self._elem_colors = self.pwr.elem_vecs_array[self._elems].copy()

        self.observation_space = Box(low=0, high=255, shape=(self._world_size, self._world_size, 6), dtype=np.uint8)
        self._xy_action_size = (world_size - brush_size) // grid_size + 1
        self.action_space = Discrete(max(len(self._elems), self._xy_action_size))

        self._world = None

        # The environment has three action stages:
        # 0: Set element id.
        # 1: Set the x coordinate.
        # 2: Set the y coordinate.
        self._action_step = 0  # The current stage.
        self._action_elem_id = None  # The selected element id.
        self._action_x = None  # The selected x coordinate.
        self._action_y = None  # The selected y coordinate

        if self._mode == 'task':
            # Set task goals.
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)
            self.cur_goal_world = None

    def set_tasks(self):
        def add_square(action_seq, elem_name, x, y, size):
            """Add a square to the action sequence."""
            for i in range(0, size):
                action_seq.append((elem_name, x + i, y + size - 1))
            for i in range(size - 2, -1, -1):
                action_seq.append((elem_name, x, y + i))
            for i in range(size - 2, -1, -1):
                action_seq.append((elem_name, x + size - 1, y + i))
            for i in range(1, size - 1):
                action_seq.append((elem_name, x + i, y))

        if self._num_elems == 2:
            # Task 1: Plant.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            self.task_infos.append(dict(task_name='task1_plant', action_seq=action_seq, tol=32))

            # Task 2: Stone.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('stone', x, y))
            self.task_infos.append(dict(task_name='task2_stone', action_seq=action_seq, tol=32))

            # Task 3: Square.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            add_square(action_seq, 'stone', 1, 1, 6)
            self.task_infos.append(dict(task_name='task3_square', action_seq=action_seq, tol=32))

            # Task 4: Four squares.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('stone', x, y))
            for sx, sy in [(0, 0), (0, 5), (5, 0), (5, 5)]:
                add_square(action_seq, 'plant', sx, sy, 3)
            self.task_infos.append(dict(task_name='task4_four_squares', action_seq=action_seq, tol=32))

            # Task 5: Mosaic.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in reversed(range(8)):
                for x in range(8):
                    if (x + y) % 2 == 0:
                        action_seq.append(('stone', x, y))
            self.task_infos.append(dict(task_name='task5_mosaic', action_seq=action_seq, tol=32))
        elif self._num_elems == 5:
            # Task 1: Squares.
            action_seq = []
            add_square(action_seq, 'plant', 1, 1, 6)
            add_square(action_seq, 'sand', 0, 0, 8)
            add_square(action_seq, 'stone', 2, 2, 4)
            add_square(action_seq, 'water', 3, 3, 2)
            self.task_infos.append(dict(task_name='task1_squares', action_seq=action_seq, tol=32))

            # Task 2: Water plant.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('water', x, y))
            add_square(action_seq, 'plant', 0, 0, 8)
            self.task_infos.append(dict(task_name='task2_water_plant', action_seq=action_seq, tol=64))

            # Task 3: Sandpile.
            action_seq = []
            add_square(action_seq, 'stone', 0, 0, 8)
            for _ in range(32):
                action_seq.extend(
                    [
                        ('sand', 3, 1),
                        ('sand', 4, 1),
                    ]
                )
            self.task_infos.append(dict(task_name='task3_sandpile', action_seq=action_seq, tol=64))

            # Task 4: Two rooms.
            action_seq = []
            for x in range(0, 8):
                action_seq.append(('plant', x, 6))
            for x in range(0, 8):
                action_seq.append(('plant', x, 7))
            for y in range(7, -1, -1):
                action_seq.append(('stone', 0, y))
            for y in range(7, -1, -1):
                action_seq.append(('stone', 7, y))
            for x in range(1, 7):
                action_seq.append(('stone', x, 4))
            for x in range(1, 7):
                action_seq.append(('stone', x, 3))
            add_square(action_seq, 'water', 2, 0, 3)
            add_square(action_seq, 'water', 3, 0, 3)
            for _ in range(4):
                action_seq.append(('fire', 3, 7))
                action_seq.append(('fire', 4, 7))
            self.task_infos.append(dict(task_name='task4_two_rooms', action_seq=action_seq, tol=64))

            # Task 5: Elements.
            action_seq = []
            for y in reversed(range(8)):
                for x in range(8):
                    action_seq.append(('plant', x, y))
            for y in [4, 7]:
                for x in range(8):
                    action_seq.append(('water', x, y))
            for _ in range(2):
                for x in range(8):
                    action_seq.append(('fire', x, 0))
            self.task_infos.append(dict(task_name='task5_elements', action_seq=action_seq, tol=96))
        elif self._num_elems == 8:
            # Task 1: Bubbles.
            action_seq = []
            for y in range(7, -1, -1):
                for x in range(8):
                    action_seq.append(('sand', x, y))
            for x in range(8):
                action_seq.append(('sand', x, 0))
            for x in range(8):
                action_seq.append(('water', x, 7))
            for x in range(8):
                action_seq.append(('gas', x, 7))
            for x in range(8):
                action_seq.append(('water', x, 7))
            self.task_infos.append(dict(task_name='task1_bubbles', action_seq=action_seq, tol=96))

            # Task 2: Firework.
            action_seq = []
            add_square(action_seq, 'wood', 0, 0, 8)
            add_square(action_seq, 'plant', 1, 1, 6)
            add_square(action_seq, 'gas', 2, 2, 4)
            for _ in range(3):
                for x in range(8):
                    action_seq.append(('fire', x, 0))
            self.task_infos.append(dict(task_name='task2_firework', action_seq=action_seq, tol=96))

            # Task 3: Three rooms.
            action_seq = []
            for x in range(8):
                action_seq.append(('ice', x, 0))
            for y in range(7, -1, -1):
                action_seq.append(('stone', 2, y))
            for y in range(7, -1, -1):
                action_seq.append(('stone', 5, y))
            for y in range(7, 0, -1):
                action_seq.append(('water', 3, y))
                action_seq.append(('water', 4, y))
            action_seq.append(('plant', 3, 3))
            action_seq.append(('plant', 4, 3))
            action_seq.append(('plant', 3, 4))
            action_seq.append(('plant', 4, 4))
            for y in range(7, 0, -1):
                action_seq.append(('gas', 0, y))
                action_seq.append(('gas', 1, y))
                action_seq.append(('gas', 6, y))
                action_seq.append(('gas', 7, y))
            self.task_infos.append(dict(task_name='task3_three_rooms', action_seq=action_seq, tol=96))

            # Task 4: Four squares.
            action_seq = []
            add_square(action_seq, 'plant', 1, 4, 3)
            add_square(action_seq, 'wood', 4, 4, 3)
            add_square(action_seq, 'ice', 1, 1, 3)
            add_square(action_seq, 'plant', 4, 1, 3)
            for _ in range(10):
                add_square(action_seq, 'plant', 4, 1, 3)
            self.task_infos.append(dict(task_name='task4_four_squares', action_seq=action_seq, tol=64))

            # Task 5: Ice plant.
            action_seq = []
            for y in range(7, -1, -1):
                for x in range(8):
                    action_seq.append(('water', x, y))
            add_square(action_seq, 'plant', 3, 3, 2)
            for _ in range(4):
                add_square(action_seq, 'stone', 0, 0, 8)
            add_square(action_seq, 'ice', 3, 3, 2)
            self.task_infos.append(dict(task_name='task5_ice_plant', action_seq=action_seq, tol=96))
        else:
            raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        if self._mode == 'task':
            # Set the task goal.
            if options is None:
                options = {}

            if 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            render_goal = False
            if 'render_goal' in options:
                render_goal = options['render_goal']

        # Initialize a new world.
        np_world = np.zeros((1, self._world_size, self._world_size), dtype=np.uint8)
        np_world[:, 0, :] = 1
        np_world[:, -1, :] = 1
        np_world[:, :, 0] = 1
        np_world[:, :, -1] = 1

        self._world = self.pw.np_to_pw(np_world).copy()
        self._action_step = 0
        self._action_elem_id = None
        self._action_x = None
        self._action_y = None

        if self._mode == 'task':
            # Before doing the actual reset, we simulate the goal actions to get a goal observation.
            self._mode = 'internal'  # Set the mode to 'internal' to prevent the agent from computing the success.
            self.reset()
            for semantic_action in self.cur_task_info['action_seq']:
                for _ in range(3):
                    self.step(self.semantic_action_to_action(*semantic_action))
            self._mode = 'task'
            goal = self._get_ob()
            self.cur_goal_world = self._world[0, 0].copy()
            if render_goal:
                goal_rendered = self.render()

            # Do the actual reset.
            self._mode = 'internal'  # Set the mode to 'internal' to prevent the agent from computing the success.
            self.reset()
            semantic_action = self.sample_semantic_action()
            # Perform a single random action (3 steps) to add some randomness to the initial state.
            for _ in range(3):
                self.step(self.semantic_action_to_action(*semantic_action))
            self._mode = 'task'

        ob = self._get_ob()
        info = dict()

        if self._mode == 'task':
            info['goal'] = goal
            if render_goal:
                info['goal_rendered'] = goal_rendered

        return ob, info

    def step(self, action):
        if self._action_step == 0:
            # Step 0: Set the element id.
            if action < len(self._elems):
                self._action_elem_id = action
            else:
                # If the action is invalid, randomly select an element.
                self._action_elem_id = np.random.randint(len(self._elems))
        elif self._action_step == 1:
            # Step 1: Set the x coordinate.
            if action < self._xy_action_size:
                self._action_x = action
            else:
                # If the action is invalid, randomly select an x coordinate.
                self._action_x = np.random.randint(self._xy_action_size)
        else:
            # Step 2: Set the y coordinate.
            if action < self._xy_action_size:
                self._action_y = action
            else:
                # If the action is invalid, randomly select a y coordinate.
                self._action_y = np.random.randint(self._xy_action_size)

            # Step the world.
            self._world = self.pw.forward(self._world)

            # Apply the action.
            np_action_world = np.zeros((1, self._world_size, self._world_size), dtype=np.uint8)
            elem = self._elems[self._action_elem_id]
            real_x = self._action_x * self._grid_size
            real_y = self._action_y * self._grid_size
            np_action_world[:, real_y : real_y + self._brush_size, real_x : real_x + self._brush_size] = elem
            world_delta = self.pw.np_to_pw(np_action_world)
            self._world = interp(
                ~self.pw.get_bool(world_delta, 'empty') & ~self.pw.get_bool(self._world, 'wall'),
                self._world,
                world_delta,
            )

            # Reset the action step.
            self._action_elem_id = None
            self._action_x = None
            self._action_y = None

        self._action_step = (self._action_step + 1) % 3

        ob = self._get_ob()
        reward = 0.0
        terminated = False
        info = dict()

        if self._mode == 'task':
            # Check if the current world matches the goal. To have some tolerance, for each pixel in the goal, we check
            # if the current world has a matching pixel that is shifted by up to one pixel in any direction. We then
            # compute the error as the number of pixels that do not match, and consider the task successful if the error
            # is below a certain threshold.
            cur_world = self._world[0, 0].copy()
            world_shifts = []
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                world_shifts.append(np.roll(cur_world, (dy, dx), axis=(0, 1)))
            world_shifts = np.stack(world_shifts, axis=0)
            match = (self.cur_goal_world == world_shifts).any(axis=0)
            error = ~match

            success = error.sum() < self.cur_task_info['tol']
            if success:
                terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

        return ob, reward, terminated, False, info

    def semantic_action_to_action(self, elem_name, x, y):
        """Convert a semantic action to an action based on the current action step."""
        elem_id = self._elem_names.index(elem_name)
        if self._action_step == 0:
            return elem_id
        elif self._action_step == 1:
            return x
        else:
            return y

    def sample_action(self):
        """Sample a random valid action."""
        if self._action_step == 0:
            return np.random.randint(len(self._elems))
        else:
            return np.random.randint(self._xy_action_size)

    def sample_semantic_action(self):
        """Sample a random semantic action."""
        elem_name = np.random.choice(self._elem_names)
        x = np.random.randint(self._xy_action_size)
        y = np.random.randint(self._xy_action_size)
        return elem_name, x, y

    def render(self):
        ob = self._get_ob()
        frame = ob[..., :3]
        frame = Image.fromarray(frame)
        frame = frame.resize((self._render_width, self._render_height), Image.NEAREST)
        frame = np.array(frame)

        return frame

    def _get_ob(self):
        world_frame = self.pwr.render(self._world).copy()  # (world_size, world_size, 3)

        # Compute an additional action frame to show the currently selected action.
        action_frame = np.zeros_like(world_frame)  # (world_size, world_size, 3)
        if self._action_step == 1:
            # Color the entire frame with the selected element color.
            color = self._elem_colors[self._action_elem_id]
            action_frame[..., :] = (color * 255.0).astype(np.uint8)
        elif self._action_step == 2:
            # Color the selected x coordinate with the selected element color.
            color = self._elem_colors[self._action_elem_id]
            real_x = self._action_x * self._grid_size
            action_frame[:, real_x : real_x + self._brush_size, :] = (color * 255.0).astype(np.uint8)
        return np.concatenate([world_frame, action_frame], axis=2)  # (world_size, world_size, 6)
