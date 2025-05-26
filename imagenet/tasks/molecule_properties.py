import torch
from typing import List, Union
from .utils import check_grad_fn, rescale_grad, ban_requires_grad

class MoleculePropertyGuidance:

    def __init__(self, args):

        from ..networks.egnn.EGNN import EGNN_energy_QM9
        from ..networks.egnn.energy import EnergyDiffusion
        from ..networks.qm9.datasets_config import get_dataset_info
        
        self.args = args

        dataset_info = get_dataset_info(args.args_gen.dataset, args.args_gen.remove_h)
        in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
        dynamics_in_node_nf = in_node_nf + 1

        net_energy = EGNN_energy_QM9(
            in_node_nf=dynamics_in_node_nf, context_node_nf=self.args.args_en.context_node_nf,
            n_dims=3, device=self.args.device, hidden_nf=self.args.args_en.nf,
            act_fn=torch.nn.SiLU(), n_layers=self.args.args_en.n_layers,
            attention=self.args.args_en.attention, tanh=self.args.args_en.tanh,
            mode=self.args.args_en.model, norm_constant=self.args.args_en.norm_constant,
            inv_sublayers=self.args.args_en.inv_sublayers, sin_embedding=self.args.args_en.sin_embedding,
            normalization_factor=self.args.args_en.normalization_factor,
            aggregation_method=self.args.args_en.aggregation_method)

        guidance = EnergyDiffusion(
            dynamics=net_energy,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=self.args.args_en.diffusion_steps,
            noise_schedule=self.args.args_en.diffusion_noise_schedule,
            noise_precision=self.args.args_en.diffusion_noise_precision,
            norm_values=self.args.args_en.normalize_factors,
            include_charges=self.args.args_en.include_charges
        )

        energy_state_dict = torch.load(args.energy_path, map_location='cpu')
        guidance.load_state_dict(energy_state_dict)

        self.classifier = guidance.to(args.device)

        ban_requires_grad(self.classifier)

    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        if check_grad:
            check_grad_fn(x_need_grad)

        x = post_process(func(x_need_grad))
        
        # classifier returns log prob!
        log_probs = self.classifier(x, **kwargs)

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)
        
    