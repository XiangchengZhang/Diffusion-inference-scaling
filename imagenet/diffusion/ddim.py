import functools
import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List, Any, Union, Tuple
import numpy as np

from diffusers.utils.torch_utils import randn_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
import logger

from tasks.networks.qm9 import dataset
from tasks.networks.qm9.utils import compute_mean_mad
from tasks.networks.qm9.datasets_config import get_dataset_info
from tasks.networks.qm9.models import DistributionProperty, DistributionNodes
from tasks.networks.egnn.EDM import EDM
from tasks.networks.egnn.EGNN import EGNN_dynamics_QM9
from tasks.networks.egnn.utils import assert_correctly_masked, assert_mean_zero_with_mask


class ImageSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(ImageSampler, self).__init__(args)
        self.object_size = (3, args.image_size, args.image_size)
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)
        self.target = args.target

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)

    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        if 'openai' in args.model_name_or_path:
            from .unet.openai import get_diffusion
        else: 
            from .unet.huggingface import get_diffusion
        
        self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = get_diffusion(args)
    

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance, global_verifier:BaseGuidance=None):
        
        tot_samples = []
        tot_compute = 0
        # n_batchs = math.ceil(sample_size / self.per_sample_batch_size)
        n_batchs = sample_size

        for batch_id in tqdm(range(n_batchs)):
            
            self.args.batch_id = batch_id

            x = randn_tensor(
                shape=(self.per_sample_batch_size, *self.object_size),
                generator=self.generator,
                device=self.device,
            )

            for t in range(self.inference_steps):
                
                tot_compute += x.shape[0]
                x = guidance.guide_step(
                    x, t, self.unet,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta,
                    global_guider=global_verifier,
                )

                # we may want to log some trajs
                if self.log_traj:
                    logger.log_samples(self.tensor_to_obj(x), fname=f'traj/time={t}')

            tot_samples.append(x)
        avg_compute = tot_compute / len(tot_samples)
        return torch.concat(tot_samples), {'compute': avg_compute}
        
    @staticmethod
    def tensor_to_obj(x):

        images = (x / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images

    @staticmethod
    def obj_to_tensor(objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images)
        return tensor_images * 2 - 1


class MoleculeSampler(BaseSampler):
    """
        This class is responsible for sampling molecules using DDIM.

    """
    def __init__(self, args: Arguments):

        super(MoleculeSampler, self).__init__(args)
        self.object_size = None                         # size of tensor shape
        self.inference_steps = args.inference_steps     # number of steps to run the diffusion model
        self.eta = args.eta                             # eta in DDIM paper
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)
        self.per_sample_batch_size = args.per_sample_batch_size

        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)


    @staticmethod
    def _get_dataloader(args_gen):
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
        return dataloaders

    @staticmethod
    def _get_generator(model_path, dataloaders, device, args, property_norms):
        dataset_info = get_dataset_info(args.args_gen.dataset, args.args_gen.remove_h)
        model, nodes_dist, prop_dist = MoleculeSampler._get_model(
            args.args_gen, device, dataset_info, dataloaders['train'],args.target)
        model_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)

        if prop_dist is not None:
            prop_dist.set_normalizer(property_norms)
        return model.to(device), nodes_dist, prop_dist, dataset_info

    @staticmethod
    def _get_model(args, device, dataset_info, dataloader_train, target):
        histogram = dataset_info['n_nodes']
        in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
        # in_node_nf: the numbder of atom type
        nodes_dist = DistributionNodes(histogram)

        # pass corresponding property into the pre-defined function
        prop_dist = DistributionProperty(dataloader_train, [target]) if target else None
        
        dynamics_in_node_nf = in_node_nf + 1
    
        net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=3, device=device, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

        vdm = EDM(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges,
        )

        return vdm, nodes_dist, prop_dist

    def _build_diffusion(self, args):
        '''
            Different diffusion models should be registered here
        '''
        # dataloader
        args.args_gen.load_charges = False
        dataloaders = self._get_dataloader(args.args_gen)
        property_norms = compute_mean_mad(dataloaders, [args.target], args.args_gen.dataset)
        # mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

        # load conditional EDM and property prediction model
        edm, nodes_dist, prop_dist, dataset_info = self._get_generator(
            args.generators_path, dataloaders,
            args.device, args, property_norms,
        )
        self.dataset_info = dataset_info
        self.device = args.device
        self.args_gen = args.args_gen
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.target = args.target

        self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = edm.get_scheduler_params(args)
        self.diffusion: EDM = edm

    def remove_mean_with_mask(self, x, node_mask):
        # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
        # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'

        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x, dim=1, keepdim=True) / N
        x = x - mean * node_mask
        return x

    def noise_fn(self, x, sigma, node_mask, **kwargs):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
            assert len(size) == 3
            x = torch.randn(size, device=device)

            x_masked = x * node_mask

            # This projection only works because Gaussian is rotation invariant around
            # zero and samples are independent!
            x_projected = self.remove_mean_with_mask(x_masked, node_mask)
            return x_projected
    
        def sample_gaussian_with_mask(size, device, node_mask):
            x = torch.randn(size, device=device)
            x_masked = x * node_mask
            return x_masked


        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(x.shape[0], x.shape[1], self.diffusion.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(x.shape[0], x.shape[1], self.diffusion.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)

        zs = z * sigma + x
        return torch.cat([
            self.remove_mean_with_mask(zs[:, :, :self.diffusion.n_dims],node_mask),
            zs[:, :, self.diffusion.n_dims:]], dim=2
        )

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):

        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        for _ in range(n_batchs):

            nodesxsample = self.nodes_dist.sample(self.per_sample_batch_size)
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)

            max_n_nodes = self.dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

            assert int(torch.max(nodesxsample)) <= max_n_nodes
            batch_size = len(nodesxsample)

            node_mask = torch.zeros(batch_size, max_n_nodes)
            for i in range(batch_size):
                node_mask[i, 0:nodesxsample[i]] = 1

            # Compute edge_mask

            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(self.device)
            node_mask = node_mask.unsqueeze(2).to(self.device)

            # # TODO FIX: This conditioning just zeros.
            # if context is None:
            #     context = self.prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(self.device) * node_mask

            n_samples = batch_size
            n_nodes = max_n_nodes

            z = self.diffusion.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
            noise_pred_func = functools.partial(self.diffusion.forward, node_mask=node_mask, edge_mask=edge_mask)

            for t in tqdm(range(self.inference_steps), total=self.inference_steps):
                
                z = guidance.guide_step(
                    z, t, noise_pred_func,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    target=context,
                )
                # z = z * node_mask
                x = z[..., :self.diffusion.n_dims]
                # project x back to the mean zero space, using the mean computed only on the nodes
                # x = x - x.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True) * node_mask
                # assert_correctly_masked(x, node_mask)
                # assert_mean_zero_with_mask(x, node_mask)

                # we may want to log some trajs
                if self.log_traj:
                    logger.log_samples(self.tensor_to_obj(z), fname=f'traj/time={t}')

            nan_mask = torch.isnan(z.reshape(z.shape[0], -1)).any(dim=1)
            logger.log(f"generate {nan_mask.shape[0]} samples, {nan_mask.sum()} of them are NaNs.")
            if nan_mask.float().mean() >= 0.5:
                logger.log(f'Warning: {nan_mask.float().mean()} of the samples are NaNs.')
                raise ValueError('Too many NaNs in the samples. Drop the run')
            z, node_mask, context = z[~nan_mask], node_mask[~nan_mask], context[~nan_mask]
            edge_mask = edge_mask.view(-1, max_n_nodes, max_n_nodes, 1)[~nan_mask].view(-1, 1)
            x, h = self.diffusion.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=False)

            # x = x - x.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True) * node_mask

            # assert_correctly_masked(x, node_mask)
            # assert_mean_zero_with_mask(x, node_mask)

            one_hot = h['categorical']
            charges = h['integer']

            # assert_correctly_masked(one_hot.float(), node_mask)
            # if self.args_gen.include_charges:
            #     assert_correctly_masked(charges.float(), node_mask)

            context = context[:, 0]  # [B, 1]
            context = context * self.prop_dist.normalizer[self.target]['mad'] + self.prop_dist.normalizer[self.target]['mean']

            tot_samples.append((one_hot, charges, x, node_mask, context))
        
        return [torch.cat([tot_samples[batch_id][_] for batch_id in range(len(tot_samples))], dim=0)
                for _ in range(len(tot_samples[0]))]

    @staticmethod
    def tensor_to_obj(x: Union[torch.Tensor, List[torch.Tensor]]):
        """
            Given a tensor represneting a batch of molecules, convert it to a list of molecule objects
        """
        one_hot, charges, x, node_mask, target = x  # [B, N, 5], [B, N, 0], [B, N, 3], [B, N], [B, 1]
        one_hot = one_hot.detach().cpu().numpy()
        charges = charges.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        node_mask = node_mask.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # list of (one_hot, charges, x, node_mask), for each molecule
        return [(one_hot[_], charges[_], x[_], node_mask[_], target[_]) for _ in range(one_hot.shape[0])]

    @staticmethod
    def obj_to_tensor(objs: List[Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
            convert a list of molecule objects into torch tensors that are used for sampling
        """
        batched_one_hot = np.stack([objs[_][0] for _ in range(len(objs))], axis=0)
        batched_one_hot = torch.from_numpy(batched_one_hot)
        batched_charges = np.stack([objs[_][1] for _ in range(len(objs))], axis=0)
        batched_charges = torch.from_numpy(batched_charges)
        batched_x = np.stack([objs[_][2] for _ in range(len(objs))], axis=0)
        batched_x = torch.from_numpy(batched_x)
        batched_node_mask = np.stack([objs[_][3] for _ in range(len(objs))], axis=0)
        batched_node_mask = torch.from_numpy(batched_node_mask)
        batched_target = np.stack([objs[_][4] for _ in range(len(objs))], axis=0)
        batched_target = torch.from_numpy(batched_target)

        return [batched_one_hot, batched_charges, batched_x, batched_node_mask, batched_target]
