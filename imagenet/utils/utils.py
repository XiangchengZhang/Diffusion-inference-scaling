import torch
import os
from typing import Union
from transformers import HfArgumentParser

from .configs import Arguments

from evaluations.image import ImageEvaluator
# from evaluations.molecule import MoleculeEvaluator
# from evaluations.audio import AudioEvaluator

from diffusion.ddim import ImageSampler, MoleculeSampler
from diffusion.audio_diffusion import AudioDiffusionSampler
from diffusion.stable_diffusion import StableDiffusionSampler

from methods.mpgd import MPGDGuidance
from methods.lgd import LGDGuidance
from methods.base import BaseGuidance
from methods.ugd import UGDGuidance
from methods.freedom import FreedomGuidance
from methods.dps import DPSGuidance
from methods.tfg import TFGGuidance
from methods.cg import ClassifierGuidance


import pickle


def get_logging_dir(arg_dict: dict):
    if arg_dict['guidance_name'] == 'tfg':
        # record rho, mu, sigma with scheduler
        suffix = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
    else:
        suffix = "guidance_strength=" + str(arg_dict['guidance_strength'])
    
    if 'bfs' in arg_dict['guidance_name']:
        suffix1 = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
        suffix2 = f"start={arg_dict['start']}+step_size={arg_dict['step_size']}"
        suffix3 = f"particles={arg_dict['per_sample_batch_size']}+temp={arg_dict['temp']}"
        suffix = f"{suffix1}/{suffix2}/{suffix3}"
    
    return os.path.join(
        arg_dict['logging_dir'],
        f"guidance_name={arg_dict['guidance_name']}+recur_steps={arg_dict['recur_steps']}+iter_steps={arg_dict['iter_steps']}",
        "model=" + arg_dict['model_name_or_path'].replace("/", '_'),
        "verifier=" + arg_dict['guide_network'].replace('/', '_'),
        "global_verifier=" + arg_dict['global_verifier'].replace('/', '_'),
        "target=" + str(arg_dict['target']).replace(" ", "_"),
        suffix,
    )

def get_config(add_logger=True) -> Arguments:
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)

    if add_logger:
        from logger import setup_logger
        args.logging_dir = get_logging_dir(vars(args))
        print("logging to", args.logging_dir)
        setup_logger(args)
    
    if args.data_type == 'molecule':
        # load args
        def _get_args_gen(args_path, argse_path):
            with open(args_path, 'rb') as f:
                args_gen = pickle.load(f)
            assert args_gen.dataset == 'qm9_second_half'

            with open(argse_path, 'rb') as f:
                args_en = pickle.load(f)

            # Add missing args!
            if not hasattr(args_gen, 'normalization_factor'):
                args_gen.normalization_factor = 1
            if not hasattr(args_gen, 'aggregation_method'):
                args_gen.aggregation_method = 'sum'

            return args_gen, args_en

        args.args_gen, args.args_en = _get_args_gen(args.args_generators_path, args.args_energy_path)
    
    # examine combined guidance

    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')

    assert len(args.tasks) == len(args.guide_networks) == len(args.targets)

    return args


def get_evaluator(args):

    if args.data_type == 'image':
        return ImageEvaluator(args)
    elif args.data_type == 'molecule':
        return MoleculeEvaluator(args)
    elif args.data_type == 'text2image':
        return ImageEvaluator(args)
    elif args.data_type == 'audio':
        return AudioEvaluator(args)
    else:
        raise NotImplementedError

def get_guidance(args, network):
    noise_fn = getattr(network, 'noise_fn', None)
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'mpgd':
        return MPGDGuidance(args, noise_fn=noise_fn)
    elif 'ugd' in args.guidance_name:
        return UGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'freedom':
        return FreedomGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'dps':
        return DPSGuidance(args, noise_fn=noise_fn)
    elif 'lgd' in args.guidance_name:
        return LGDGuidance(args, noise_fn=noise_fn)
    elif "tfg" in args.guidance_name:
        return TFGGuidance(args, noise_fn=noise_fn)
    elif 'cg' in args.guidance_name:
        return ClassifierGuidance(args, noise_fn=noise_fn)
    elif 'bfs' in args.guidance_name:
        from methods.bfs import BFSGuidance
        return BFSGuidance(args, noise_fn=noise_fn)
    else:
        raise NotImplementedError

def get_network(args):
    
    if args.data_type == 'image':
        return ImageSampler(args)
    elif args.data_type == 'molecule':
        return MoleculeSampler(args)
    elif args.data_type == 'text2image':
        return StableDiffusionSampler(args)
    elif args.data_type == 'audio':
        return AudioDiffusionSampler(args)
    else:
        raise NotImplementedError
