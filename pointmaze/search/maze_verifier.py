import torch
from diffuser import datasets


def distance_to_closest_non_adjacent_wall_boundary(pos, wall_pos, wall_size, adjacency):
    lower_bound = wall_pos - wall_size   # (2,)
    upper_bound = wall_pos + wall_size   # (2,)

    is_inside = torch.logical_and(
        torch.logical_and(pos[..., 0] >= lower_bound[0], pos[..., 0] <= upper_bound[0]),
        torch.logical_and(pos[..., 1] >= lower_bound[1], pos[..., 1] <= upper_bound[1])
    )
    dist_x_low = torch.where(
        is_inside,
        pos[..., 0] - lower_bound[0],
        torch.zeros_like(pos[..., 0])
    ) if adjacency[2] == 1 else torch.full_like(pos[..., 0], float('inf'))
    dist_x_high = torch.where(
        is_inside,
        upper_bound[0] - pos[..., 0],
        torch.zeros_like(pos[..., 0])
    ) if adjacency[3] == 1 else torch.full_like(pos[..., 0], float('inf'))
    dist_y_low = torch.where(
        is_inside,
        pos[..., 1] - lower_bound[1],
        torch.zeros_like(pos[..., 1])
    ) if adjacency[0] == 1 else torch.full_like(pos[..., 1], float('inf'))
    dist_y_high = torch.where(
        is_inside,
        upper_bound[1] - pos[..., 1],
        torch.zeros_like(pos[..., 1])
    ) if adjacency[1] == 1 else torch.full_like(pos[..., 1], float('inf'))
    dist = torch.minimum(
        torch.minimum(dist_x_low, dist_x_high),
        torch.minimum(dist_y_low, dist_y_high)
    )
    return dist

def batched_inside_wall_loss_non_adjacent(positions, wall_boxes, wall_adjacency):
    barch_size, timesteps, _ = positions.shape
    device = positions.device
    all_distances = torch.zeros((barch_size, timesteps, len(wall_boxes)), device=device)
    for w, (wall_pos, wall_size) in enumerate(wall_boxes):
        pos_2d = positions.reshape(barch_size * timesteps, 2)
        dist_2d = distance_to_closest_non_adjacent_wall_boundary(pos_2d, wall_pos.to(device), wall_size.to(device), wall_adjacency[w])
        dist_2d = dist_2d.view(barch_size, timesteps)
        all_distances[..., w] = dist_2d
    dist = all_distances.max(dim=-1).values
    loss = dist ** 2
    return loss


def get_wall_boxes_and_adjacency(env,device):
    wall_boxes = []
    wall_adjacency = []
    if env._maze_type == 'ultra':
        consider_ball = True
    else:
        consider_ball = False
    for i in range(env.maze_map.shape[0]):
        for j in range(env.maze_map.shape[1]):
            if env.maze_map[i, j] == 1:
                wall_pos = (j * env._maze_unit - env._offset_x, i * env._maze_unit - env._offset_y)
                wall_size = (env._maze_unit / 2, env._maze_unit / 2)
                if consider_ball:
                    wall_size = (wall_size[0] + 0.7, wall_size[1] + 0.7)
                assert wall_pos == env.ij_to_xy((i, j))
                wall_boxes.append((wall_pos, wall_size))
                adjacency = [1] * 4
                if i > 0 and env.maze_map[i-1, j] == 1 or i == 0:
                    adjacency[0] = 0
                if i < env.maze_map.shape[0]-1 and env.maze_map[i+1, j] == 1 or i == env.maze_map.shape[0]-1:
                    adjacency[1] = 0
                if j > 0 and env.maze_map[i, j-1] == 1 or j == 0:
                    adjacency[2] = 0
                if j < env.maze_map.shape[1]-1 and env.maze_map[i, j+1] == 1 or j == env.maze_map.shape[1]-1:
                    adjacency[3] = 0
                if adjacency[0] == 0 and adjacency[1] == 0 and adjacency[2] == 0 and adjacency[3] == 0:
                    if i == 0: adjacency[1] = 1
                    elif i == env.maze_map.shape[0]-1: adjacency[0] = 1
                    elif j == 0: adjacency[3] = 1
                    elif j == env.maze_map.shape[1]-1: adjacency[2] = 1
                    else: raise ValueError("Invalid adjacency")

                wall_adjacency.append(adjacency)
    
    # Convert wall_boxes into torch tensors
    wall_boxes = [(torch.tensor(pos[:2], dtype=torch.float32,device=device), torch.tensor(size[:2], dtype=torch.float32,device=device)) for pos, size in wall_boxes]
    return wall_boxes,wall_adjacency



from search.utils import check_grad_fn, rescale_grad, ban_requires_grad
from search.configs import Arguments

class MazeVerifier():
    def __init__(self, args: Arguments = Arguments()):
        self.env = datasets.load_environment(args.dataset)
        device = torch.device(args.device)
        self.wall_boxes, self.adjacency = get_wall_boxes_and_adjacency(self.env,device=device)
        self.device = device

    def get_guidance(self, x, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        if check_grad:
            check_grad_fn(x)
        
        x = post_process(func(x))
        
        x = x[..., 2:4] # only use the x, y coordinates

        log_probs = batched_inside_wall_loss_non_adjacent(x, self.wall_boxes, self.adjacency) * -1

        if return_logp:
            return log_probs.sum(dim=tuple(range(1, log_probs.ndim)))

        grad = torch.autograd.grad(log_probs.mean(), x)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)






