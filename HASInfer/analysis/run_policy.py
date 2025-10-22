#!/usr/bin/env python3

"""
Unified interface for running policy models with support for different caching modes.

Usage examples:
# Without caching (original mode)
python -m diffusion_policy.activation_utils.run_policy -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization/square_ph/original -d cuda:0 --demo_idx 0 --cache_mode original

# Using threshold caching
python -m diffusion_policy.activation_utils.run_policy -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization/square_ph/threshold_5 -d cuda:0 --demo_idx 0 --cache_mode threshold --cache_threshold 5

# Using optimal steps caching
python -m diffusion_policy.activation_utils.run_policy -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization/square_ph/optimal_cosine -d cuda:0 --demo_idx 0 --cache_mode optimal --optimal_steps_dir assets/square_ph/optimal_steps/cosine --metric cosine --num_caches 5 --bu

# Using Fix mode caching (all FFN blocks share optimal steps from the last layer)
python -m diffusion_policy.activation_utils.run_policy -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization/square_ph/fix -d cuda:0 --demo_idx 0 --cache_mode fix --optimal_steps_dir assets/square_ph/optimal_steps/cosine --metric cosine --num_caches 5

# Using Propagate mode caching (first FFN block only computes at step 0, other steps are cached)
python -m diffusion_policy.activation_utils.run_policy -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization/square_ph/propagate -d cuda:0 --demo_idx 0 --cache_mode propagate

# Using Edit mode caching (all blocks share custom steps)
python -m diffusion_policy.activation_utils.run_policy -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization/square_ph/edit -d cuda:0 --demo_idx 0 --cache_mode edit --edit_steps 0,20,40,60,80
"""
import sys
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import pathlib
import click
import hydra
import torch
import dill
import logging
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global function to set random seed
def set_seed_for_policy(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner
    cudnn.deterministic = True

def run_policy(
    checkpoint, 
    output_dir, 
    device, 
    demo_idx,
    cache_mode='original',
    cache_threshold=5, 
    optimal_steps_dir=None, 
    num_caches=30, 
    metric='cosine',
    num_bu_blocks=3,
    edit_steps=None,
    interpolation_ratio=1.0,
    reference_activations_path=None,
    return_obs_action=False,
    random_seed=11  # Add random seed parameter
):
    """
    Run policy model and return model, inputs, and optional outputs.

    Args:
        checkpoint: Model checkpoint path
        output_dir: Output directory
        device: Device to run on
        demo_idx: Demonstration index
        cache_mode: Caching mode ('original', 'threshold', 'optimal', 'fix', 'propagate', 'edit')
        cache_threshold: Cache threshold, update cache every N steps
        optimal_steps_dir: Optimal steps directory path
        num_caches: Number of cache updates
        metric: Similarity metric type
        num_bu_blocks: Number of blocks to apply BU algorithm to, disable BU when 0
        edit_steps: Custom steps to use in edit mode
        interpolation_ratio: Interpolation ratio between cached and original activations (default=1.0)
        reference_activations_path: Path to reference (original) activations for interpolation
        return_obs_action: Whether to return observations and actions
        random_seed: Random seed for reproducibility

    Returns:
        policy: Policy model
        obs_dict: Observation data
        action_dict: Action data if return_obs_action is True
    """
    # Set random seed
    set_seed_for_policy(random_seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # Initialize workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Get policy model
    device = torch.device(device)
    policy = workspace.model
    if hasattr(cfg.training, 'use_ema') and cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # Apply cache acceleration
    if cache_mode != 'original':
        logger.info(f"Applying {cache_mode} mode cache acceleration")

        if cache_mode in ['optimal', 'fix']:
            logger.info(f"Optimal steps directory: {optimal_steps_dir}, metric: {metric}, num_caches: {num_caches}")
        elif cache_mode == 'threshold':
            logger.info(f"Cache threshold: {cache_threshold}")
        elif cache_mode == 'edit':
            logger.info(f"Edit mode custom steps: {edit_steps}")

        if num_bu_blocks > 0:
            logger.info(f"Using BU algorithm to optimize {num_bu_blocks} blocks")

        if interpolation_ratio < 1.0:
            logger.info(f"Using interpolation ratio: {interpolation_ratio}")
            if reference_activations_path:
                logger.info(f"Reference activation path: {reference_activations_path}")

        # Cache configuration parameters
        cache_params = {
            'cache_mode': cache_mode
        }

        if cache_mode == 'threshold':
            cache_params['cache_threshold'] = cache_threshold
        elif cache_mode in ['optimal', 'fix', 'random']:
            cache_params['optimal_steps_dir'] = optimal_steps_dir
            cache_params['num_caches'] = num_caches
            cache_params['metric'] = metric
        elif cache_mode == 'edit':
            cache_params['edit_steps'] = edit_steps

        # Add BU algorithm parameters
        cache_params['num_bu_blocks'] = num_bu_blocks

        # Add interpolation parameters
        cache_params['interpolation_ratio'] = interpolation_ratio
        if reference_activations_path:
            cache_params['reference_activations_path'] = reference_activations_path

        # Apply caching
        policy = FastDiffusionPolicy.apply_cache(policy, **cache_params)
    else:
        logger.info("Using original mode, no cache acceleration applied")
    
    # Modify environment runner configuration
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    if demo_idx < 10000:
        env_runner_cfg['n_train'] = 1
        env_runner_cfg['n_train_vis'] = 0
        # Check if environment runner uses train_start_idx or train_start_seed parameter
        if 'train_start_idx' in env_runner_cfg:
            env_runner_cfg['train_start_idx'] = demo_idx
        elif 'train_start_seed' in env_runner_cfg:
            env_runner_cfg['train_start_seed'] = demo_idx
        env_runner_cfg['n_test'] = 0
        env_runner_cfg['n_test_vis'] = 0
    else:
        env_runner_cfg['n_train'] = 0
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test'] = 1
        env_runner_cfg['n_test_vis'] = 0
        env_runner_cfg['test_start_seed'] = demo_idx
    env_runner_cfg['n_envs'] = 1

    # Create environment runner
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        output_dir=str(output_dir))

    env = env_runner.env
    this_init_fns = [env_runner.env_init_fn_dills[0 if demo_idx < 10000 else -1]]
    env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

    # Reset environment
    obs = env.reset()
    policy.reset()

    # If using cache, ensure cache policy is reset
    if cache_mode != 'original' and hasattr(policy, 'reset_cache'):
        policy.reset_cache()

    # Prepare observation data
    # Environments may return observations in different formats, need to handle
    np_obs_dict = {}
    if isinstance(obs, dict):
        np_obs_dict = obs
    elif isinstance(obs, (list, tuple)):
        try:
            # Some environments return (obs_dict, reward, done, info) format
            np_obs_dict = dict(obs[0])
        except (ValueError, TypeError) as e:
            # If above handling fails, log error and try other approaches
            logger.warning(f"Error handling obs: {e}")
            logger.warning(f"obs type: {type(obs)}, content: {obs}")
            # If unable to handle, try simple processing
            if len(obs) > 0 and isinstance(obs[0], dict):
                np_obs_dict = obs[0]
            else:
                raise ValueError(f"Unable to handle returned observation format: {type(obs)}, {obs}")
    elif isinstance(obs, np.ndarray):
        # Handle NumPy array type observations
        logger.info(f"Detected NumPy array observation, shape: {obs.shape}")
        # Assume this is low-dimensional observation, put directly into field named 'obs'
        np_obs_dict = {'obs': obs}
    else:
        raise ValueError(f"Unsupported observation format: {type(obs)}")
    
    if hasattr(env_runner, 'past_action') and env_runner.past_action:
        past_action = np.zeros((1, env_runner.n_obs_steps-1, env_runner.env_meta['action_dim']))
        np_obs_dict['past_action'] = past_action.astype(np.float32)

    # Transfer to device
    obs_dict = dict_apply(np_obs_dict,
        lambda x: torch.from_numpy(x).to(device=device))

    # If need to return action, perform prediction
    if return_obs_action:
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        return policy, obs_dict, action_dict

    return policy, obs_dict

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Model checkpoint path')
@click.option('-o', '--output_dir', required=True, help='Output directory')
@click.option('-d', '--device', default='cuda:0', help='Device to run on')
@click.option('--demo_idx', default=0, type=int, help='Demonstration index')
@click.option('--cache_mode', default='original',
              type=click.Choice(['original', 'threshold', 'optimal', 'fix', 'propagate', 'edit']),
              help='Caching mode')
@click.option('--cache_threshold', default=5, type=int, help='Cache threshold, update cache every N steps')
@click.option('--optimal_steps_dir', default=None, help='Optimal steps directory path')
@click.option('--num_caches', default=30, type=int, help='Number of cache updates')
@click.option('--metric', default='cosine', help='Similarity metric type, used to load optimal steps file')
@click.option('--num_bu_blocks', default=3, type=int, help='Number of blocks to apply BU algorithm to, disable BU when 0')
@click.option('--edit_steps', default=None, help='Edit mode custom steps (comma-separated integers, e.g., "0,20,40,60,80")')
@click.option('--interpolation_ratio', default=1.0, type=float, help='Interpolation ratio between cached and original activations (default=1.0)')
@click.option('--reference_activations_path', default=None, help='Path to reference (original) activations for interpolation')
@click.option('--random_seed', default=11, type=int, help='Random seed for reproducibility')
def main(checkpoint, output_dir, device, demo_idx,
         cache_mode, cache_threshold, optimal_steps_dir,
         num_caches, metric, num_bu_blocks, edit_steps,
         interpolation_ratio, reference_activations_path, random_seed):

    # Process edit_steps parameter
    processed_edit_steps = None
    if edit_steps is not None:
        try:
            processed_edit_steps = [int(x.strip()) for x in edit_steps.split(',')]
            logger.info(f"Parsed edit steps: {processed_edit_steps}")
        except Exception as e:
            logger.error(f"Error parsing edit_steps '{edit_steps}': {e}")
            logger.error("Format should be comma-separated integers, e.g., '0,20,40,60,80'")
            return
    
    policy, obs_dict, action_dict = run_policy(
        checkpoint, output_dir, device, demo_idx,
        cache_mode, cache_threshold, optimal_steps_dir,
        num_caches, metric, num_bu_blocks,
        processed_edit_steps, interpolation_ratio, reference_activations_path,
        return_obs_action=True,
        random_seed=random_seed
    )

    # Transfer back to CPU
    np_action_dict = dict_apply(action_dict,
        lambda x: x.detach().to('cpu').numpy())

    action = np_action_dict['action']
    if not np.all(np.isfinite(action)):
        raise RuntimeError("Nan or Inf action")

    logger.info(f"Action shape: {action.shape}")
    logger.info(f"Action: {action}")

    return action

if __name__ == '__main__':
    main() 