import numpy as np


class Behavior:
    """Base class for action behaviors."""

    def __init__(self, env):
        self._env = env
        self._done = False
        self._step = 0

        self._size = self._env.unwrapped._world_size // self._env.unwrapped._brush_size
        assert self._env.unwrapped._brush_size == self._env.unwrapped._grid_size
        self._elem_name = None
        self._sequence = None

    @property
    def done(self):
        return self._done

    def reset(self, ob, info):
        pass

    def select_action(self, ob, info):
        x, y = self._sequence[self._step]

        self._step += 1
        if self._step == len(self._sequence):
            self._done = True

        return self._elem_name, x, y


class FillBehavior(Behavior):
    """Fill the entire grid with a single element."""

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._elem_name = np.random.choice(self._env.unwrapped._elem_names)

        # Randomly flip the fill directions.
        flip_x = np.random.randint(2)
        flip_y = np.random.randint(2)
        flip_xy = np.random.randint(2)

        self._sequence = []
        for i in range(self._size * self._size):
            x = i % self._size
            if flip_x:
                x = self._size - x - 1
            y = i // self._size
            if flip_y:
                y = self._size - y - 1
            if flip_xy:
                x, y = y, x

            self._sequence.append((x, y))


class LineBehavior(Behavior):
    """Fill a single line with a single element."""

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._elem_name = np.random.choice(self._env.unwrapped._elem_names)

        # Randomly select the line direction.
        target_idx = np.random.randint(self._size)
        flip_dir = np.random.randint(2)
        flip_xy = np.random.randint(2)

        self._sequence = []
        for i in range(self._size):
            x, y = i, target_idx
            if flip_dir:
                y = self._size - 1 - y
            if flip_xy:
                x, y = y, x

            self._sequence.append((x, y))


class SquareBehavior(Behavior):
    """Draw a square with a single element."""

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._elem_name = np.random.choice(self._env.unwrapped._elem_names)

        length = np.random.randint(1, self._size)
        x1 = np.random.randint(self._size - length)
        x2 = x1 + length
        y1 = np.random.randint(self._size - length)
        y2 = y1 + length

        sides = []
        sides.append([(x1, y) for y in range(y1, y2 + 1)])
        sides.append([(x2, y) for y in range(y1, y2 + 1)])
        sides.append([(x, y1) for x in range(x1, x2 + 1)])
        sides.append([(x, y2) for x in range(x1, x2 + 1)])

        # Randomly reverse sides.
        for i in range(4):
            if np.random.randint(2):
                sides[i].reverse()

        # Randomly shuffle the order of sides.
        np.random.shuffle(sides)

        self._sequence = []
        for side in sides:
            self._sequence.extend(side)
