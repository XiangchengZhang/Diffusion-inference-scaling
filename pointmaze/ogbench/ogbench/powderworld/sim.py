"""A numpy version of Powderworld simulator.

The code is based on the original Powderworld simulator written in PyTorch. It is supposed to work identically to the
original simulator, but for performance reasons, the update rules that are not used in the OGBench Powderworld tasks are
commented out. To use the full simulator, uncomment the unused update rules in the `register_update_rules` function.
"""

from collections import namedtuple

import numpy as np

Info = namedtuple('Info', ['rand_movement', 'rand_interact', 'rand_element'])

# ================ REGISTER ELEMENTS. =================
# Name:    ID, Density, GravityInter
pw_elements = {
    'empty': (0, 1, 1),
    'wall': (1, 4, 0),
    'sand': (2, 3, 1),
    'water': (3, 2, 1),
    'gas': (4, 0, 1),
    'wood': (5, 4, 0),
    'ice': (6, 4, 0),
    'fire': (7, 0, 1),
    'plant': (8, 4, 0),
    'stone': (9, 3, 1),
    'lava': (10, 3, 1),
    'acid': (11, 2, 1),
    'dust': (12, 2, 1),
    'cloner': (13, 4, 0),
    'agentFish': (14, 2, 1),
    'agentBird': (15, 4, 0),
    'agentKangaroo': (16, 3, 1),
    'agentMole': (17, 3, 1),
    'agentLemming': (18, 3, 1),
    'agentSnake': (19, 4, 0),
    'agentRobot': (20, 3, 1),
}

pw_element_names = [
    'empty',
    'wall',
    'sand',
    'water',
    'gas',
    'wood',
    'ice',
    'fire',
    'plant',
    'stone',
    'lava',
    'acid',
    'dust',
    'cloner',
    'agentFish',
    'agentBird',
    'agentKangaroo',
    'agentMole',
    'agentLemming',
    'agentSnake',
    'agentRobot',
]


# ================================================
# ============= HELPERS ==================
# ================================================


def get_below(x):
    return np.roll(x, shift=-1, axis=2)


def get_above(x):
    return np.roll(x, shift=1, axis=2)


def get_left(x):
    return np.roll(x, shift=1, axis=3)


def get_right(x):
    return np.roll(x, shift=-1, axis=3)


def get_in_cardinal_direction(x, directions):
    y = get_right(x) * (directions == 0)
    y = y + get_below(x) * (directions == 2)
    y = y + get_left(x) * (directions == 4)
    y = y + get_above(x) * (directions == 6)
    return y


def interp(switch, if_false, if_true):
    return (~switch) * if_false + (switch) * if_true


def interp_int(switch, if_false, if_true: int):
    return (~switch) * if_false + (switch) * if_true


def interp2(switch_a, switch_b, if_false, if_a, if_b):
    return ((~switch_a) & (~switch_b)) * if_false + (switch_a) * if_a + (switch_b) * if_b


def interp_swaps8(swaps, world, w0, w1, w2, w3, w4, w5, w6, w7):
    new_world = world * (swaps == -1)
    new_world += w0 * (swaps == 0)
    new_world += w1 * (swaps == 1)
    new_world += w2 * (swaps == 2)
    new_world += w3 * (swaps == 3)
    new_world += w4 * (swaps == 4)
    new_world += w5 * (swaps == 5)
    new_world += w6 * (swaps == 6)
    new_world += w7 * (swaps == 7)
    return new_world


def interp_swaps4(swaps, world, w0, w1, w2, w3):
    new_world = world * (swaps == -1)
    new_world += w0 * (swaps == 0)
    new_world += w1 * (swaps == 1)
    new_world += w2 * (swaps == 2)
    new_world += w3 * (swaps == 3)
    return new_world


def normalize(x, p=2, axis=0, eps=1e-12):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / (norm + eps)


# Source: https://github.com/99991/NumPyConv2D
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
    """
    Applies a 2D convolution to an array of images. Technically, this function
    computes a correlation instead of a convolution since the kernels are not
    flipped.

    input: numpy array of images with shape = (n, c, h, w)
    weight: numpy array with shape = (c_out, c // groups, kernel_height, kernel_width)
    bias: numpy array of shape (c_out,), default None
    stride: step width of convolution kernel, int or (int, int) tuple, default 1
    padding: padding around images, int or (int, int) tuple or "same", default 0
    dilation: spacing between kernel elements, int or (int, int) tuple, default 1
    groups: split c and c_out into groups to reduce memory usage (must both be divisible), default 1
    padding_mode: "zeros", "reflect", "replicate", "circular", or whatever np.pad supports, default "zeros"

    This function is indended to be similar to PyTorch's conv2d function.
    For more details, see:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    """
    c_out, c_in_by_groups, kh, kw = weight.shape
    kernel_size = (kh, kw)

    if isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(dilation, int):
        dilation = [dilation, dilation]

    if padding:
        input = conv2d_pad(input, padding, padding_mode, stride, dilation, kernel_size)

    n, c_in, h, w = input.shape
    dh, dw = dilation
    sh, sw = stride
    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1
    assert c_in % groups == 0, f'Number of input channels ({c_in}) not divisible by groups ({groups}).'
    assert c_out % groups == 0, f'Number of output channels ({c_out}) not divisible by groups ({groups}).'
    c_in_group = c_in // groups
    c_out_group = c_out // groups
    kernel_shape = (c_in_group, dilated_kh, dilated_kw)

    input = input.reshape(n, groups, c_in_group, h, w)
    weight = weight.reshape(groups, c_out_group, c_in_by_groups, kh, kw)

    # Cut out kernel-shaped regions from input
    windows = np.lib.stride_tricks.sliding_window_view(input, kernel_shape, axis=(-3, -2, -1))

    # Apply stride and dilation. Prepare for broadcasting to handle groups.
    windows = windows[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
    weight = weight[np.newaxis, :, :, np.newaxis, np.newaxis, :, :, :]
    h_out, w_out = windows.shape[3:5]

    # Dot product equivalent to either of the next two lines but 10 times faster
    # y = np.einsum("abcdeijk,abcdeijk->abcde", windows, weight)
    # y = (windows * weight).sum(axis=(-3, -2, -1))
    windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
    weight = weight.reshape(1, groups, c_out_group, 1, 1, c_in_group * kh * kw)
    y = np.einsum('abcdei,abcdei->abcde', windows, weight)

    # Concatenate groups as output channels
    y = y.reshape(n, c_out, h_out, w_out)

    if bias is not None:
        y = y + bias.reshape(1, c_out, 1, 1)

    return y


def conv2d_pad(input, padding, padding_mode, stride, dilation, kernel_size):
    if padding == 'valid':
        return input

    if padding == 'same':
        h, w = input.shape[-2:]
        sh, sw = stride
        dh, dw = dilation
        kh, kw = kernel_size
        ph = (h - 1) * (sh - 1) + (kh - 1) * dh
        pw = (w - 1) * (sw - 1) + (kw - 1) * dw
        ph0 = ph // 2
        ph1 = ph - ph0
        pw0 = pw // 2
        pw1 = pw - pw0
    else:
        if isinstance(padding, int):
            padding = [padding, padding]
        ph0, pw0 = padding
        ph1, pw1 = padding

    pad_width = ((0, 0), (0, 0), (ph0, ph1), (pw0, pw1))

    mode = {
        'zeros': 'constant',
        'reflect': 'reflect',
        'replicate': 'edge',
        'circular': 'wrap',
    }.get(padding_mode, padding_mode)

    return np.pad(input, pad_width, mode)


# ================================================
# ============= GENERAL CLASS =================
# ================================================


class PWSim:
    def __init__(self):
        self.elements = pw_elements
        self.element_names = pw_element_names

        self.NUM_ELEMENTS = len(self.elements)
        # [ElementID(0), Density(1), GravityInter(2), VelocityField(3, 4), Color(5), Custom1(6), Custom2(7), Custom3(8)]
        # Custom 1              Custom 2          Custom 3
        # BirdVelX              BirdVelY
        #                                         DidGravity
        # FluidMomentum
        # FluidMomentum         KangarooJump
        # MoleDirection
        # SnakeDirection.       SnakeEnergy
        self.NUM_CHANNEL = 1 + 1 + 1 + 2 + 1 + 3
        self.pw_type = np.float32

        # ================ NUMPY KERNELS =================
        self.elem_vecs = {}
        self.elem_vecs_array = np.zeros((self.NUM_ELEMENTS, self.NUM_CHANNEL), dtype=self.pw_type)
        for elem_name, elem in self.elements.items():
            elem_vec = np.zeros(self.NUM_CHANNEL)
            elem_vec[0] = elem[0]
            elem_vec[1] = elem[1]
            elem_vec[2] = elem[2]
            self.elem_vecs[elem_name] = elem_vec[None, :, None, None]
            self.elem_vecs_array[elem[0]] = elem_vec

        self.neighbor_kernel = np.ones((1, 1, 3, 3), dtype=self.pw_type)
        self.zero = np.zeros((1, 1), dtype=self.pw_type)
        self.one = np.ones((1, 1), dtype=self.pw_type)

        self.up = np.array([-1, 0])[None, :, None, None]
        self.down = np.array([1, 0])[None, :, None, None]
        self.left = np.array([0, -1])[None, :, None, None]
        self.right = np.array([0, 1])[None, :, None, None]

        self.register_update_rules()

    # ==================================================
    # ============ REGISTER UPDATE RULES ===============
    # ==================================================
    def register_update_rules(self):
        """
        Overwrite this function with your own set of update rules to change behavior.
        """
        self.update_rules = [
            BehaviorStone(self),
            # BehaviorMole(self),
            BehaviorGravity(self),
            BehaviorSand(self),
            # BehaviorLemming(self),
            BehaviorFluidFlow(self),
            BehaviorIce(self),
            BehaviorWater(self),
            BehaviorFire(self),
            BehaviorPlant(self),
            # BehaviorLava(self),
            # BehaviorAcid(self),
            # BehaviorCloner(self),
            # BehaviorFish(self),
            # BehaviorBird(self),
            # BehaviorKangaroo(self),
            # BehaviorSnake(self),
            BehaviorVelocity(self),
        ]  # Unused rules commented out for performance
        self.update_rules_jit = None

    # =========== WORLD EDITING HELPERS ====================
    def add_element(self, world_slice, element_name, wind=None):
        if isinstance(element_name, int):
            element_name = self.element_names[element_name]

        if element_name == 'wind':
            world_slice[:, 3:5] = wind
        else:
            world_slice[...] = self.elem_vecs[element_name]
            if element_name == 'agentSnake':
                world_slice[:, 7] = 1

    def add_element_rc(self, world_slice, rr, cc, element_name):
        if isinstance(element_name, int):
            element_name = self.element_names[element_name]
        world_slice[:, :, rr, cc] = self.elem_vecs[element_name]

    def id_to_pw(self, world_ids):
        world = self.elem_vecs_array[world_ids]
        world = np.transpose(world, (0, 3, 1, 2))
        return world

    def np_to_pw(self, np_world):
        np_world_ids = np_world.astype(int).copy()
        return self.id_to_pw(np_world_ids)

    # =========== UPDATE HELPERS ====================
    def get_elem(self, world, elemname):
        elem_id = self.elements[elemname][0]
        return (world[:, 0:1] == elem_id).astype(self.pw_type)

    def get_bool(self, world, elemname):
        elem_id = self.elements[elemname][0]
        return world[:, 0:1] == elem_id

    def direction_func(self, d, x):
        if d == 0:
            return get_right(x)
        elif d == 1:
            return get_right(get_below(x))
        elif d == 2:
            return get_below(x)
        elif d == 3:
            return get_left(get_below(x))
        elif d == 4:
            return get_left(x)
        elif d == 5:
            return get_left(get_above(x))
        elif d == 6:
            return get_above(x)
        elif d == 7:
            return get_right(get_above(x))

    def forward(self, world):
        # Helper Functions
        rand_movement = np.random.rand(world.shape[0], 1, world.shape[2], world.shape[3]).astype(
            self.pw_type
        )  # For gravity
        rand_interact = np.random.rand(world.shape[0], 1, world.shape[2], world.shape[3]).astype(
            self.pw_type
        )  # For gravity
        rand_element = np.random.rand(world.shape[0], 1, world.shape[2], world.shape[3]).astype(
            self.pw_type
        )  # For gravity

        info = (rand_movement, rand_interact, rand_element)

        for update_rule in self.update_rules:
            world = update_rule.forward(world, info)

        return world


# ================================================
# ============== RENDERER ========================
# ================================================
class PWRenderer:
    def __init__(self):
        pw_type = np.float32
        self.elem_vecs_array = np.zeros((len(pw_elements), 3), dtype=pw_type)
        self.elem_vecs_array = (
            np.array(
                [
                    [236, 240, 241],  # EMPTY #ECF0F1
                    [108, 122, 137],  # WALL #6C7A89
                    [243, 194, 58],  # SAND #F3C23A
                    [75, 119, 190],  # WATER #4B77BE
                    [179, 157, 219],  # GAS #875F9A
                    [202, 105, 36],  # WOOD #CA6924
                    [137, 196, 244],  # ICE #89C4F4
                    [249, 104, 14],  # FIRE #F9680E
                    [38, 194, 129],  # PLANT #26C281
                    [38, 67, 72],  # STONE #264348
                    [157, 41, 51],  # LAVA #9D2933
                    [176, 207, 120],  # ACID #B0CF78
                    [255, 179, 167],  # DUST #FFB3A7
                    [191, 85, 236],  # CLONER #BF55EC
                    [0, 229, 255],  # AGENT FISH #00E5FF
                    [61, 90, 254],  # AGENT BIRD #3D5AFE
                    [121, 85, 72],  # AGENT KANGAROO #795548
                    [56, 142, 60],  # AGENT MOLE #388E3C
                    [158, 157, 36],  # AGENT LEMMING #9E9D24
                    [198, 40, 40],  # AGENT SNAKE #C62828
                    [224, 64, 251],  # AGENT ROBOT #E040FB
                ],
                dtype=pw_type,
            )
            / 255.0
        )

        self.vector_color_kernel = np.array([200, 100, 100], dtype=np.float32)
        self.vector_color_kernel /= 255.0
        self.vector_color_kernel = self.vector_color_kernel[:, None, None]

    # ================ RENDERING ====================
    def forward(self, world):
        img = self.elem_vecs_array[world[0:1, 0].astype(int)][0].transpose(2, 0, 1)
        if world.shape[1] > 1:
            velocity_field = world[0, 3:5]
            velocity_field_magnitudes = np.linalg.norm(velocity_field, axis=0)[None]

            velocity_field_angles_raw = (1 / (2 * np.pi)) * np.arccos(
                velocity_field[1] / (velocity_field_magnitudes + 0.001)
            )
            is_y_lessthan_zero = velocity_field[0] < 0
            velocity_field_angles_raw = interp(
                switch=is_y_lessthan_zero,
                if_false=velocity_field_angles_raw,
                if_true=(1 - velocity_field_angles_raw),
            )
            velocity_field_angles = velocity_field_angles_raw

            velocity_field_colors = self.vector_color_kernel

            velocity_field_display = np.clip(velocity_field_magnitudes / 5, 0, 0.5)
            img = (1 - velocity_field_display) * img + velocity_field_display * velocity_field_colors
        img = np.clip(img, 0, 1)
        return img

    def render(self, world):
        img = self.forward(world)
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        return img


# =========================================================================
# ====================== CORE UPDATE BEHAVIORS ============================
# =========================================================================


class BehaviorGravity:
    """
    Run gravity procedure.
    Loop through each possible density (1-5).
    In kernel form, compute for each block:
        IF density == currDensity && density BELOW is less && both gravity-affected -> Become below.
        (Else)If density ABOVE is greater && density ABOVE == currDensity && both gravity-affected -> Become above.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        world[:, 8:9] = interp(switch=(world[:, 2:3] == 1), if_false=world[:, 8:9], if_true=self.pw.zero)

        above = get_above(world)
        below = get_below(world)

        is_density_below_less = below[:, 1:2] - world[:, 1:2] < 0
        is_gravity = world[:, 2:3] == 1
        is_gravity_below = below[:, 2:3] == 1

        is_center_and_below_gravity = is_gravity_below & is_gravity
        does_become_below = is_density_below_less & is_center_and_below_gravity
        does_become_above = get_above(does_become_below)

        has_overlap = does_become_below & does_become_above
        does_become_below_real = does_become_below & ~has_overlap
        does_become_above_real = get_above(does_become_below_real)

        world[:] = interp2(
            switch_a=does_become_below_real, switch_b=does_become_above_real, if_false=world, if_a=below, if_b=above
        )
        world[:, 8:9] = interp(switch=does_become_above_real, if_false=world[:, 8:9], if_true=self.pw.one)

        return world


class BehaviorSand:
    """
    Run sand-piling procedure.
    Loop over each piling block type. In kernel form, for each block:
        If dir=left and BELOW_LEFT density is less && both gravity-affected -> Become below-left.
        If ABOVE_RIGHT dir=left and ABOVE_RIGHT density is less && both gravity-affected -> Become above-right.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        fall_dir = rand_movement > 0.5
        not_did_gravity = ~(world[:, 8:9] > 0)
        for fallLeft in [True, False]:
            get_in_dir = get_left if fallLeft else get_right
            get_in_not_dir = get_right if fallLeft else get_left
            world_below_left = get_in_dir(get_below(world))
            world_above_right = get_in_not_dir(get_above(world))

            fall_dir = (rand_movement > 0.5) if fallLeft else (rand_movement <= 0.5)
            rand_above_right = get_in_not_dir(get_above(fall_dir))

            is_below_left_density_lower = (world[:, 1:2] - world_below_left[:, 1:2]) > 0
            is_above_right_density_higher = (world_above_right[:, 1:2] - world[:, 1:2]) > 0
            is_gravity = world[:, 2:3] == 1
            is_below_left_gravity = world_below_left[:, 2:3] == 1
            is_above_right_gravity = world_above_right[:, 2:3] == 1
            is_element = self.pw.get_bool(world, 'sand') | self.pw.get_bool(world, 'dust')
            is_above_right_element = self.pw.get_bool(world_above_right, 'sand') | self.pw.get_bool(
                world_above_right, 'dust'
            )

            is_matching_fall = fall_dir
            is_above_right_matching_fall = rand_above_right
            not_did_gravity = ~(world[:, 8:9] > 0)
            not_did_gravity_below_left = ~(world_below_left[:, 8:9] > 0)
            not_did_gravity_above_right = ~(world_above_right[:, 8:9] > 0)

            does_become_below_left = (
                is_element
                & not_did_gravity_below_left
                & is_matching_fall
                & is_below_left_density_lower
                & is_below_left_gravity
                & not_did_gravity
            )
            does_become_above_right = (
                is_above_right_element
                & not_did_gravity_above_right
                & is_above_right_matching_fall
                & is_above_right_density_higher
                & is_above_right_gravity
                & not_did_gravity
            )

            world[:] = interp2(
                switch_a=does_become_below_left,
                switch_b=does_become_above_right,
                if_false=world,
                if_a=world_below_left,
                if_b=world_above_right,
            )
        return world


class BehaviorStone:
    """Run stone-stability procedure. If a stone is next to two stones, turn gravity off. Otherwise, turn it on."""

    def __init__(self, pw):
        self.pw = pw
        self.stone_kernel = np.zeros((1, 1, 3, 3), dtype=self.pw.pw_type)
        self.stone_kernel[0, 0, 0, 0] = 1
        self.stone_kernel[0, 0, 0, 2] = 1

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        stone = self.pw.get_elem(world, 'stone')
        has_stone_supports = conv2d(stone, self.stone_kernel, padding=1)
        world[:, 2:3] = (1 - stone) * world[:, 2:3] + stone * (has_stone_supports < 2)
        return world


class BehaviorFluidFlow:
    """
    Run fluid-flowing procedure. Same as sand-piling, but move LEFT/RIGHT instead of BELOW-LEFT/BELOW-RIGHT.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        new_fluid_momentum = np.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw.pw_type)
        for fallLeft in [True, False]:
            get_in_dir = get_left if fallLeft else get_right
            get_in_not_dir = get_right if fallLeft else get_left
            fall_dir = (rand_movement + world[:, 6:7] + new_fluid_momentum) > 0.5
            is_matching_fall = fall_dir if fallLeft else (~fall_dir)
            world_left = get_in_dir(world)
            world_right = get_in_not_dir(world)

            is_air_move = self.pw.get_bool(world, 'agentKangaroo') | self.pw.get_bool(world, 'agentLemming')
            is_element = (
                self.pw.get_bool(world, 'empty')
                | self.pw.get_bool(world, 'water')
                | self.pw.get_bool(world, 'gas')
                | self.pw.get_bool(world, 'lava')
                | self.pw.get_bool(world, 'acid')
                | is_air_move
            )
            is_left_density_lower = (world[:, 1:2] - world_left[:, 1:2]) > 0
            is_gravity = world[:, 2:3] == 1
            is_left_gravity = world_left[:, 2:3] == 1
            not_did_gravity_left = ~(world[:, 8:9] > 0) | is_air_move

            does_become_left = (
                is_matching_fall
                & is_element
                & not_did_gravity_left
                & is_left_density_lower
                & is_left_gravity
                & is_gravity
            )
            does_become_right = get_in_not_dir(does_become_left)

            has_overlap = does_become_left & does_become_right
            does_become_left_real = does_become_left & ~has_overlap
            does_become_right_real = get_in_not_dir(does_become_left_real)

            new_fluid_momentum += does_become_right_real * (2 if fallLeft else -2)

            world[:] = interp2(
                switch_a=does_become_left_real,
                switch_b=does_become_right_real,
                if_false=world,
                if_a=world_left,
                if_b=world_right,
            )

        world[:, 6:7, :, :] = interp(
            switch=(
                self.pw.get_bool(world, 'empty')
                | self.pw.get_bool(world, 'water')
                | self.pw.get_bool(world, 'gas')
                | self.pw.get_bool(world, 'lava')
                | self.pw.get_bool(world, 'acid')
                | self.pw.get_bool(world, 'agentKangaroo')
                | self.pw.get_bool(world, 'agentLemming')
            ),
            if_false=world[:, 6:7, :, :],
            if_true=new_fluid_momentum,
        )

        return world


class BehaviorIce:
    """
    Ice melting. Ice touching water or air turns to water.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        ice_chance = rand_interact
        ice_melting_neighbors = (
            self.pw.get_elem(world, 'empty')
            + self.pw.get_elem(world, 'fire')
            + self.pw.get_elem(world, 'lava')
            + self.pw.get_elem(world, 'water')
        )
        ice_can_melt = conv2d(ice_melting_neighbors, self.pw.neighbor_kernel, padding=1) > 1
        does_turn_water = self.pw.get_bool(world, 'ice') & ice_can_melt & (ice_chance < 0.02)
        world[:] = interp(switch=does_turn_water, if_false=world, if_true=self.pw.elem_vecs['water'])
        return world


class BehaviorWater:
    """
    Water freezing. Water touching 3+ ices can turn to ice.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        ice_chance = rand_element
        water_can_freeze = conv2d(self.pw.get_elem(world, 'ice'), self.pw.neighbor_kernel, padding=1) >= 3
        does_turn_ice = self.pw.get_bool(world, 'water') & water_can_freeze & (ice_chance < 0.05)
        world[:] = interp(switch=does_turn_ice, if_false=world, if_true=self.pw.elem_vecs['ice'])
        return world


class BehaviorFire:
    """
    Fire burning.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        burn_chance = rand_interact
        fire_and_lava = self.pw.get_elem(world, 'fire') + self.pw.get_elem(world, 'lava')
        has_fire_neighbor = conv2d(fire_and_lava, self.pw.neighbor_kernel, padding=1) > 0
        does_burn_wood = self.pw.get_bool(world, 'wood') & (burn_chance < 0.05)
        does_burn_bird = self.pw.get_bool(world, 'agentBird') & (burn_chance < 0.05)
        does_burn_plant = self.pw.get_bool(world, 'plant') & (burn_chance < 0.2)
        does_burn_agent = (
            self.pw.get_bool(world, 'agentFish')
            | self.pw.get_bool(world, 'agentLemming')
            | self.pw.get_bool(world, 'agentKangaroo')
            | self.pw.get_bool(world, 'agentMole')
        ) & (burn_chance < 0.2)
        does_burn_gas = self.pw.get_bool(world, 'gas') & (burn_chance < 0.2)
        does_burn_dust = self.pw.get_bool(world, 'dust')
        does_burn_ice = self.pw.get_bool(world, 'ice') & (burn_chance < 0.2) & has_fire_neighbor
        does_burn = (
            does_burn_wood | does_burn_plant | does_burn_gas | does_burn_dust | does_burn_bird | does_burn_agent
        ) & has_fire_neighbor

        # Velocity for fire
        world[:, 3:5] -= 8 * get_left(does_burn & has_fire_neighbor) * self.pw.left
        world[:, 3:5] -= 8 * get_above(does_burn & has_fire_neighbor) * self.pw.up
        world[:, 3:5] -= 8 * get_below(does_burn & has_fire_neighbor) * self.pw.down
        world[:, 3:5] -= 8 * get_right(does_burn & has_fire_neighbor) * self.pw.right

        world[:, 3:5] -= 30 * get_left(does_burn_dust & has_fire_neighbor) * self.pw.left
        world[:, 3:5] -= 30 * get_above(does_burn_dust & has_fire_neighbor) * self.pw.up
        world[:, 3:5] -= 30 * get_below(does_burn_dust & has_fire_neighbor) * self.pw.down
        world[:, 3:5] -= 30 * get_right(does_burn_dust & has_fire_neighbor) * self.pw.right

        world[:] = interp(switch=does_burn, if_false=world, if_true=self.pw.elem_vecs['fire'])
        world[:] = interp(switch=does_burn_ice, if_false=world, if_true=self.pw.elem_vecs['water'])

        # Fire spread. (Fire+burnable, or Lava)=> creates a probability to spread to air.
        burnables = (
            self.pw.get_elem(world, 'wood')
            + self.pw.get_elem(world, 'plant')
            + self.pw.get_elem(world, 'gas')
            + self.pw.get_elem(world, 'dust')
            + self.pw.get_bool(world, 'agentFish')
            + self.pw.get_bool(world, 'agentBird')
            + self.pw.get_bool(world, 'agentKangaroo')
            + self.pw.get_bool(world, 'agentMole')
            + self.pw.get_bool(world, 'agentLemming')
        )
        fire_with_burnable_neighbor = conv2d(burnables, self.pw.neighbor_kernel, padding=1) * fire_and_lava
        in_fire_range = conv2d(
            fire_with_burnable_neighbor + self.pw.get_elem(world, 'lava'), self.pw.neighbor_kernel, padding=1
        )
        does_burn_empty = self.pw.get_bool(world, 'empty') & (in_fire_range > 0) & (burn_chance < 0.3)
        world[:] = interp(switch=does_burn_empty, if_false=world, if_true=self.pw.elem_vecs['fire'])

        # Fire fading. Fire just has a chance to fade, if not next to a burnable neighbor.
        fire_chance = rand_element
        has_burnable_neighbor = conv2d(burnables, self.pw.neighbor_kernel, padding=1)
        does_fire_turn_empty = self.pw.get_bool(world, 'fire') & (fire_chance < 0.4) & (has_burnable_neighbor == 0)
        world[:] = interp(switch=does_fire_turn_empty, if_false=world, if_true=self.pw.elem_vecs['empty'])

        return world


class BehaviorPlant:
    """
    Plants-growing. If there is water next to plant, and < 4 neighbors, chance to grow there.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        plant_chance = rand_interact
        plant_counts = conv2d(self.pw.get_elem(world, 'plant'), self.pw.neighbor_kernel, padding=1)
        does_plantgrow = self.pw.get_bool(world, 'water') & (plant_chance < 0.05)
        does_plantgrow_plant = does_plantgrow & (plant_counts <= 3) & (plant_counts >= 1)
        does_plantgrow_empty = does_plantgrow & (plant_counts > 3)

        wood_ice_counts = conv2d(
            self.pw.get_elem(world, 'ice') + self.pw.get_elem(world, 'wood'), self.pw.neighbor_kernel, padding=1
        )
        does_plantgrow_plant = does_plantgrow_plant | (
            (wood_ice_counts > 0) & (plant_chance < 0.2) & self.pw.get_bool(world, 'empty') & (plant_counts > 0)
        )

        world[:] = interp2(
            switch_a=does_plantgrow_plant,
            switch_b=does_plantgrow_empty,
            if_false=world,
            if_a=self.pw.elem_vecs['plant'],
            if_b=self.pw.elem_vecs['empty'],
        )
        return world


class BehaviorLava:
    """
    Lava-water interaction. Lava that is touching water turns to stone.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        water_counts = conv2d(self.pw.get_elem(world, 'water'), self.pw.neighbor_kernel, padding=1)
        does_turn_stone = (water_counts > 0) & self.pw.get_bool(world, 'lava')
        world[:] = interp(switch=does_turn_stone, if_false=world, if_true=self.pw.elem_vecs['stone'])

        lava_counts = conv2d(self.pw.get_elem(world, 'lava'), self.pw.neighbor_kernel, padding=1)
        does_turn_stone = (lava_counts > 0) & self.pw.get_bool(world, 'sand')
        world[:] = interp(switch=does_turn_stone, if_false=world, if_true=self.pw.elem_vecs['stone'])
        return world


class BehaviorAcid:
    """
    Acid destroys everything except wall and cloner.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        acid_rand = rand_interact < 0.2
        is_block = ~(
            self.pw.get_bool(world, 'empty')
            | self.pw.get_bool(world, 'wall')
            | self.pw.get_bool(world, 'acid')
            | self.pw.get_bool(world, 'cloner')
            | self.pw.get_bool(world, 'agentSnake')
            | self.pw.get_bool(world, 'gas')
        )
        is_acid = self.pw.get_bool(world, 'acid')
        does_acid_dissapear = (is_acid & acid_rand & get_below(is_block)) | (is_acid & acid_rand & get_above(is_block))
        does_block_dissapear = (is_block & get_above(acid_rand) & get_above(is_acid)) | (
            is_block & get_below(acid_rand) & get_below(is_acid)
        )
        does_dissapear = does_acid_dissapear | does_block_dissapear
        world[:] = interp(switch=does_dissapear, if_false=world, if_true=self.pw.elem_vecs['gas'])
        return world


class BehaviorCloner:
    """
    Cloner keeps track of the first element it touches, and then replaces neighboring empty blocks with that element.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return self.pw.get_bool(world, 'cloner').any().item()

    def forward(self, world, info):
        cloner_assigns = world[:, 6:7, :, :]
        is_not_cloner = ~self.pw.get_bool(world, 'cloner')
        labels = world[:, 0:1]
        for get_dir in [get_below, get_above, get_left, get_right]:
            is_cloner_empty = self.pw.get_bool(world, 'cloner') & ((cloner_assigns == 0) | (cloner_assigns == 13))
            dir_labels = get_dir(labels)
            world[:, 6:7, :, :] = interp2(
                switch_a=is_not_cloner,
                switch_b=is_cloner_empty,
                if_a=cloner_assigns,
                if_b=dir_labels,
                if_false=cloner_assigns,
            )

        # Cloner produce
        cloner_assigns_ids = np.clip(world[:, 6], 0, self.pw.NUM_ELEMENTS - 1).astype(int)
        cloner_assigns_vec = self.pw.elem_vecs_array[cloner_assigns_ids]
        cloner_assigns_vec = np.transpose(cloner_assigns_vec, (0, 3, 1, 2))
        for get_dir in [get_below, get_above, get_left, get_right]:
            cloner_assigns_vec_dir = get_dir(cloner_assigns_vec)
            is_dir_cloner_not_empty = get_dir(
                self.pw.get_bool(world, 'cloner') & ((cloner_assigns != 0) & (cloner_assigns != 13))
            ) & self.pw.get_bool(world, 'empty')
            world[:] = interp(switch=is_dir_cloner_not_empty, if_false=world, if_true=cloner_assigns_vec_dir)
        return world


class BehaviorVelocity:
    """
    Velocity field movement
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return (world[:, 3:5].abs() > 0.9).any().item()

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        velocity_field = world[:, 3:5]

        for n in range(2):
            velocity_field_magnitudes = np.linalg.norm(velocity_field, axis=1)[:, None]
            velocity_field_angles_raw = (1 / (2 * np.pi)) * np.arccos(
                velocity_field[:, 1:2] / (velocity_field_magnitudes + 0.001)
            )
            is_y_lessthan_zero = velocity_field[:, 0:1] < 0
            velocity_field_angle = interp(
                switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw)
            )
            velocity_field_delta = velocity_field.copy()
            velocity_angle_int = np.remainder(np.floor(velocity_field_angle * 8 + 0.5), 8)
            is_velocity_enough = (velocity_field_magnitudes > (1.0 if n == 0 else 2.0)) & (
                ~self.pw.get_bool(world, 'wall')
            )

            dw = []
            for angle in [0, 1, 2, 3, 4, 5, 6, 7]:
                dw.append(self.pw.direction_func(angle, world))

            swaps = -np.ones((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw.pw_type)
            for angle in [0, 1, 2, 3, 4, 5, 6, 7]:
                direction_empty = self.pw.get_bool(dw[angle], 'empty')
                direction_swap = self.pw.direction_func(angle, swaps)
                match = (
                    (velocity_angle_int == angle)
                    & is_velocity_enough
                    & (swaps == -1)
                    & (direction_swap == -1)
                    & direction_empty
                )
                opposite_match = self.pw.direction_func((angle + 4) % 8, match)
                swaps = interp_int(match, swaps, angle)
                swaps = interp_int(opposite_match, swaps, (angle + 4) % 8)

            velocity_field_old = velocity_field.copy()
            world[:] = interp_swaps8(swaps, world, dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7])
            world[:, 3:5] = world[:, 3:5] * 0.5 + velocity_field_old * 0.5

        # Velocity field reduction
        velocity_field *= 0.95
        for i in range(1):
            velocity_field[:, 0:1] = (
                conv2d(velocity_field[:, 0:1], self.pw.neighbor_kernel / 18, padding=1) + velocity_field[:, 0:1] * 0.5
            )
            velocity_field[:, 1:2] = (
                conv2d(velocity_field[:, 1:2], self.pw.neighbor_kernel / 18, padding=1) + velocity_field[:, 1:2] * 0.5
            )
        world[:, 3:5] = velocity_field
        return world


class BehaviorFish:
    """
    Fish move randomly.
    IF (Fish & direction) -> Become opposite direction.
    IF in opposite direction is (Fish & direction) -> become Fish.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return self.pw.get_bool(world, 'agentFish').any().item()

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        # Small issue here: the order matters because sometimes fish move twice if they roll the correct angle.
        # We could fix it by keeping track of new fish and not allowing them to move.

        for angle in [0, 1, 2, 3]:
            is_gravity = world[:, 2:3] == 1
            is_angle_match = np.floor(rand_movement * 4) == angle
            density = world[:, 1:2]
            is_empty_in_dir = self.pw.direction_func(angle * 2, is_gravity & (density <= 2))
            is_fish = self.pw.get_bool(world, 'agentFish')
            opposite_world = self.pw.direction_func(angle * 2, world)
            does_become_opposite = (
                is_angle_match
                & is_empty_in_dir
                & is_fish
                & self.pw.direction_func(angle * 2, ~is_fish)
                & (rand_interact < 0.2)
            )
            does_become_fish = self.pw.direction_func(((angle * 2) + 4) % 8, does_become_opposite)

            world[:] = interp2(
                switch_a=does_become_fish,
                switch_b=does_become_opposite,
                if_false=world,
                if_a=self.pw.elem_vecs['agentFish'],
                if_b=opposite_world,
            )

        return world


class BehaviorBird:
    """
    Birds have a random velocity, and create velocity in that direction.
    """

    def __init__(self, pw):
        self.pw = pw
        self.obstacle_kernel = np.concatenate(
            [
                np.broadcast_to((np.arange(7) - 3)[None, None, :, None], (1, 1, 7, 7)),
                np.broadcast_to((np.arange(7) - 3)[None, None, None, :], (1, 1, 7, 7)),
            ],
            axis=0,
        ).astype(self.pw.pw_type)
        self.flocking_kernel = np.ones((2, 2, 13, 13), dtype=self.pw.pw_type)
        self.flocking_kernel[1, 0] = 0
        self.flocking_kernel[0, 1] = 0
        self.flocking_kernel[:, :, 3, 3] = 0

    def check_filter(self, world):
        return self.pw.get_bool(world, 'agentBird').any().item()

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        is_empty_bird_vel = (world[:, 6:7] == 0) & (world[:, 7:8] == 0)
        is_empty_bird = is_empty_bird_vel & self.pw.get_bool(world, 'agentBird')
        random_dirs = np.concatenate([np.cos(rand_movement * np.pi * 2), np.sin(rand_movement * np.pi * 2)], axis=1)
        bird_vel = world[:, 6:8].copy()
        bird_vel = interp(switch=is_empty_bird, if_false=bird_vel, if_true=random_dirs)

        not_empty = (~self.pw.get_bool(world, 'empty')).astype(self.pw.pw_type)
        vel_delta_obstacle = -conv2d(not_empty, self.obstacle_kernel, padding=3)
        vel_delta_flocking = 1 * conv2d(
            bird_vel * self.pw.get_elem(world, 'agentBird'), self.flocking_kernel, padding=6
        )

        bird_vel += self.pw.get_elem(world, 'agentBird') * (vel_delta_obstacle + vel_delta_flocking)
        bird_vel = normalize(bird_vel + 0.01, axis=1)

        original_68 = world[:, 6:8].copy()
        world[:, 3:5] += self.pw.get_elem(world, 'agentBird') * bird_vel
        world[:, 6:8] = interp(self.pw.get_bool(world, 'agentBird'), original_68, bird_vel)

        return world


class BehaviorKangaroo:
    """
    Kangaroos move left/right and also randomly jump and pick up blocks.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        density = world[:, 1:2]
        fluid_momentum = world[:, 6:7]
        kangaroo_jump_state = world[:, 7:8]
        is_kangaroo = self.pw.get_bool(world, 'agentKangaroo')

        is_kangaroo_jump = is_kangaroo & (rand_element < 0.05) & get_below(density >= 3)
        kangaroo_jump_state = interp(switch=is_kangaroo_jump, if_false=kangaroo_jump_state, if_true=self.pw.one)
        kangaroo_jump_state = interp(
            switch=is_kangaroo, if_false=kangaroo_jump_state, if_true=kangaroo_jump_state - 0.1
        )
        world[:, 7:8] = kangaroo_jump_state

        world[:, 3:5] += (is_kangaroo & (kangaroo_jump_state > 0)) * self.pw.up * 4

        return world


class BehaviorMole:
    """
    Moles burrow through solids.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return self.pw.get_bool(world, 'agentMole').any().item()

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        beetle_dir = world[:, 6:7]

        is_beetle = self.pw.get_bool(world, 'agentMole')
        does_beetle_dir_change = is_beetle & (rand_element < 0.1)
        new_beetle_dir = np.floor(rand_movement * 4)
        beetle_dir = interp(switch=does_beetle_dir_change, if_false=beetle_dir, if_true=new_beetle_dir)
        world[:, 6:7] = beetle_dir

        density = world[:, 1:2]
        is_beetle_num = self.pw.get_elem(world, 'agentMole')
        has_supports = conv2d((density >= 3).astype(self.pw.pw_type), self.pw.neighbor_kernel, padding=1)
        world[:, 2:3] = (1 - is_beetle_num) * world[:, 2:3] + is_beetle_num * (has_supports < 2)

        dw = []
        for angle in [0, 1, 2, 3]:
            dw.append(self.pw.direction_func(angle * 2, world))

        for angle in [0, 1, 2, 3]:
            is_angle_match = beetle_dir == angle
            is_solid_in_dir = dw[angle][:, 1:2] >= 3
            is_wall_in_dir = self.pw.get_bool(dw[angle], 'wall')
            is_empty_in_dir = self.pw.get_bool(dw[angle], 'empty')
            is_beetle_in_dor = self.pw.get_bool(dw[angle], 'agentMole')
            does_move_in_dir = ((rand_element < 0.5) & is_solid_in_dir & ~is_wall_in_dir) | (
                (rand_element < 0.1) & ~is_wall_in_dir
            )
            does_become_opposite = is_angle_match & does_move_in_dir & is_beetle & ~is_beetle_in_dor
            does_become_beetle = self.pw.direction_func(((angle * 2) + 4) % 8, does_become_opposite)

            resulting_world = interp(switch=is_solid_in_dir, if_false=dw[angle], if_true=self.pw.elem_vecs['empty'])

            world[:] = interp2(
                switch_a=does_become_beetle,
                switch_b=does_become_opposite,
                if_false=world,
                if_a=dw[(angle + 2) % 4],
                if_b=resulting_world,
            )

        return world


class BehaviorLemming:
    """
    If lemmings run into a block, they move up.
    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return True

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        fluid_momentum = world[:, 6:7, :, :]
        for fallLeft in [True, False]:
            fall_dir = fluid_momentum > 0.5
            is_matching_fall = fall_dir if fallLeft else (~fall_dir)
            get_in_dir = get_left if fallLeft else get_right
            get_in_not_dir = get_right if fallLeft else get_left
            is_element = self.pw.get_bool(world, 'agentLemming')

            density = world[:, 1:2]
            density_forward = get_in_dir(density)
            density_above = get_above(density)
            density_forward_above = get_in_dir(get_above(density))

            is_forward_density_higher = (density_forward - density) >= 0
            is_above_density_lower = (density_above - density) < 0
            is_density_forward_above_lower = (density_forward_above - density) < 0

            does_become_above = (
                is_element & is_forward_density_higher & is_above_density_lower & is_density_forward_above_lower
            )
            does_become_below = get_below(does_become_above)

            world_above = get_above(world)
            world_below = get_below(world)
            world[:] = interp2(
                switch_a=does_become_above,
                switch_b=does_become_below,
                if_false=world,
                if_a=world_above,
                if_b=world_below,
            )

        return world


class BehaviorSnake:
    """
    [s1][s2][e]
    [s=does_become_opposite -> become snake_trail]
    [e=does_become_snake -> become swap]
    Snake is working, but we need to prevent turns into itself / turns into the wall.
    How to do so? How about don't turn if there is a wall present or there is more snake present?

    """

    def __init__(self, pw):
        self.pw = pw

    def check_filter(self, world):
        return self.pw.get_bool(world, 'agentSnake').any().item()

    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info

        snake_dir = world[:, 6:7]
        was_snake = self.pw.get_bool(world, 'agentSnake').copy()
        old_snake_dir = snake_dir.copy()
        old_snake_energy = world[:, 7:8].copy()

        does_become_trail = np.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=bool)
        does_become_snake = np.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=bool)
        dir_snake_came_from = np.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw.pw_type)
        ones = np.ones((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw.pw_type)

        does_turn = rand_movement < 0.1

        # For each angle, if it's a snake, become trail.
        # If coming from behind is this angle -> does_become_this_angle = (It becomes a snake. There is a snake before entering the tile.)
        # If random, or infront is a wall, does_turn.
        # Dir_snake_come_from = which angle entering from?
        for angle in [0, 1, 2, 3]:
            is_angle_match = snake_dir == angle
            is_snake_angle = self.pw.get_bool(world, 'agentSnake') & is_angle_match
            does_become_trail = does_become_trail | is_snake_angle
            does_become_this_angle = self.pw.direction_func(((angle * 2) + 4) % 8, is_snake_angle) & ~self.pw.get_bool(
                world, 'wall'
            )
            does_turn = does_turn | (
                self.pw.direction_func(
                    angle * 2, self.pw.get_bool(world, 'wall') | self.pw.get_bool(world, 'agentSnake')
                )
                & does_become_this_angle
            )
            world[:] = interp((does_become_snake & does_become_this_angle), world, self.pw.elem_vecs['empty'])
            does_become_snake = does_become_snake | does_become_this_angle
            dir_snake_came_from = interp(
                switch=does_become_this_angle, if_false=dir_snake_came_from, if_true=ones * angle
            )

            # Perform Swaps
        acid_or_not = interp(rand_element < 0.05, self.pw.elem_vecs['empty'], self.pw.elem_vecs['acid'])
        snake_trail = interp(old_snake_energy > 0, acid_or_not, self.pw.elem_vecs['agentSnake'])
        world[:] = interp(does_become_trail, if_false=world, if_true=snake_trail)
        world[:] = interp(does_become_snake, if_false=world, if_true=self.pw.elem_vecs['agentSnake'])

        # This is where I am about to turn in.
        turned_dir_came_from = (dir_snake_came_from + 1 - 2 * (rand_element < 0.5).astype(int)) % 4
        in_dir = get_in_cardinal_direction(world, turned_dir_came_from * 2)
        # BUT, don't turn if there is a snake or wall in that direction!
        does_turn = does_turn & ~self.pw.get_bool(in_dir, 'agentSnake') & ~self.pw.get_bool(in_dir, 'wall')

        is_snake = self.pw.get_bool(world, 'agentSnake')
        dir_snake_came_from = interp(switch=does_turn, if_false=dir_snake_came_from, if_true=turned_dir_came_from)
        new_snake_dir = interp(switch=(is_snake & ~was_snake), if_false=old_snake_dir, if_true=dir_snake_came_from)

        world[:, 6:7] = interp(is_snake, world[:, 6:7], new_snake_dir)
        world[:, 7:8] = interp(is_snake, world[:, 7:8], old_snake_energy - 0.1)

        return world
