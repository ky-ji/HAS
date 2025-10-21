"""
BAC (Block-wise Adaptive Caching) Evaluation Script

Usage:
# Threshold caching
python scripts/eval_fast_diffusion_policy.py --checkpoint <ckpt> -o <out_dir> --device cuda:0 --cache_mode threshold --cache_threshold 5 --skip_video

# Optimal caching with BU algorithm
python scripts/eval_fast_diffusion_policy.py --checkpoint <ckpt> -o <out_dir> --device cuda:0 --cache_mode optimal --optimal_steps_dir <assets/.../optimal_steps/cosine> --metric cosine --num_caches 30 --num_bu_blocks 3 --skip_video

# Baseline (no caching)
python scripts/eval_fast_diffusion_policy.py --checkpoint <ckpt> -o <out_dir> --device cuda:0 --cache_mode original --skip_video

"""
import sys
import os
import pathlib
import click
import hydra
import torch
import dill
import numpy as np
import time
import json
import logging
from omegaconf import OmegaConf
from copy import deepcopy

# Import thop library for FLOPs computation
from thop import profile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import caching wrapper
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from diffusion_policy.common.pytorch_util import dict_apply

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_script")

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Model checkpoint path')
@click.option('-o', '--output_dir', required=True, help='Output directory')
@click.option('-d', '--device', default='cuda:0', help='Device to run on')
@click.option('--cache_threshold', default=5, type=int, help='Cache threshold: update cache every N steps')
@click.option('--optimal_steps_dir', default=None, help='Directory for optimal schedules (required for optimal mode)')
@click.option('--num_caches', default=30, type=int, help='Number of cache updates in optimal schedule filename')
@click.option('--metric', default='cosine', help='Similarity metric for optimal schedules')
@click.option('--cache_mode', default=None, type=click.Choice(['original', 'threshold', 'optimal']), help='Caching mode')
@click.option('--num_bu_blocks', default=0, type=int, help='Number of blocks for BU algorithm (0 disables BU)')
@click.option('--skip_video', is_flag=True, help='Skip video rendering for faster evaluation')

def main(checkpoint, output_dir, device, cache_threshold, optimal_steps_dir,
         num_caches, metric, cache_mode, num_bu_blocks, skip_video):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    logger.info(f"Configuration loaded: {cfg._target_}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    logger.info("Workspace loaded successfully")

    # Get original policy
    original_policy = workspace.model
    original_policy.to(device)
    original_policy.eval()

    # Log policy information
    has_model = hasattr(original_policy, 'model')
    logger.info(f"Policy type: {type(original_policy).__name__}")
    if has_model:
        logger.info(f"Model type: {type(original_policy.model).__name__}")

    # Get number of action frames per inference
    actions_per_inference = original_policy.n_action_steps
    logger.info(f"Number of action frames per inference: {actions_per_inference}")

    # Create fast policy instance
    logger.info("Creating policy copy for cache acceleration...")
    fast_policy = deepcopy(original_policy)
    
    # Apply cache acceleration
    if cache_mode == 'optimal':
        logger.info(f"Applying optimal steps-based cache acceleration, dir={optimal_steps_dir}, metric={metric}")
        fast_policy = FastDiffusionPolicy.apply_cache(
            policy=fast_policy,
            cache_mode='optimal',
            optimal_steps_dir=optimal_steps_dir,
            num_caches=num_caches,
            metric=metric,
            num_bu_blocks=num_bu_blocks
        )
    else:  # threshold or original
        logger.info(f"Applying threshold-based cache acceleration, threshold={cache_threshold}")
        fast_policy = FastDiffusionPolicy.apply_cache(
            policy=fast_policy,
            cache_mode=cache_mode,  # Could be None, 'threshold', or 'original'
            cache_threshold=cache_threshold,
            num_bu_blocks=num_bu_blocks
        )
    
    # Generate simulated input for inference testing
    B = 1  # Batch size
    To = cfg.n_obs_steps  # Number of observation steps
    
    # Check if it's a low-dimensional model
    if hasattr(cfg, 'shape_meta'):
        logger.info("Detected image model configuration")
        # Create tensors for each observation type
        obs_dict = {}
        for key, shape in cfg.shape_meta['obs'].items():
            # Adjust shape to fit batch size and observation steps
            tensor_shape = [B, To] + list(shape['shape'])
            obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
    else:
        logger.info("Detected low-dimensional model configuration")
        # Create low-dimensional observation input using obs_dim
        obs_dim = cfg.obs_dim if hasattr(cfg, 'obs_dim') else None
        
        # If obs_dim is not explicitly defined, try to get it from task configuration
        if obs_dim is None and hasattr(cfg, 'task'):
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            if 'obs_dim' in task_cfg:
                obs_dim = task_cfg['obs_dim']
                logger.info(f"Using obs_dim from configuration: {obs_dim}")
        
        # Create observation dictionary
        obs_dict = {
            'obs': torch.zeros((B, To, obs_dim), device=device, dtype=torch.float32)
        }
        
        # Handle possible past_action
        if hasattr(cfg, 'use_past_action') and cfg.use_past_action:
            action_dim = cfg.action_dim
            obs_dict['past_action'] = torch.zeros((B, To, action_dim), device=device, dtype=torch.float32)

    # Log information about created input
    logger.info(f"Created input dictionary with keys: {list(obs_dict.keys())}")
    for key, tensor in obs_dict.items():
        logger.info(f"  {key}: shape={tensor.shape}")
    
    # Warmup and measure original policy time
    logger.info("Warming up...")
    try:
        with torch.no_grad():
            original_policy.predict_action(obs_dict)
            #fast_policy.reset_cache()
            fast_policy.predict_action(obs_dict)
        logger.info("Warmup complete")
    except Exception as e:
        logger.error(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute FLOPs (using thop library)
    flops_value = 0.0

    logger.info("\nComputing policy FLOPs using thop...")
    try:
        # Create a copy of input before model prediction to avoid interference
        obs_dict_copy = {}
        for key, value in obs_dict.items():
            obs_dict_copy[key] = value.clone()
            
        # Prepare input parameters needed by the model
        # 1. Extract features from input to create sample and conditions
        with torch.no_grad():
            # Normalize input
            nobs = fast_policy.normalizer.normalize(obs_dict_copy)
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]
            
            # Ensure correct action_dim and obs_feature_dim are obtained
            if hasattr(fast_policy, 'action_dim'):
                Da = fast_policy.action_dim
            elif hasattr(cfg, 'action_dim'):
                Da = cfg.action_dim
            else:
                Da = cfg.task.action_dim
            
            if hasattr(fast_policy, 'obs_feature_dim'):
                Do = fast_policy.obs_feature_dim
            else:
                # For low-dimensional models without obs_feature_dim, use obs_dim
                if hasattr(cfg, 'obs_dim'):
                    Do = cfg.obs_dim
                else:
                    Do = cfg.task.obs_dim
            
            logger.info(f"Action dimension: {Da}, Observation feature dimension: {Do}")
            
            # Create necessary inputs
            device = fast_policy.device
            dtype = torch.float32
            timestep = torch.zeros(B, dtype=torch.long, device=device)  # Initial timestep is 0
            
            # Handle conditional input
            cond = None
            sample = None
            
            # Get action dimension and observation dimension
            action_dim = fast_policy.action_dim
            obs_feature_dim = fast_policy.obs_dim if hasattr(fast_policy, 'obs_dim') else Do

            logger.info(f"Action dimension: {action_dim}, Observation feature dimension: {obs_feature_dim}")
            
            # Check policy type and handle accordingly
            if hasattr(fast_policy, 'obs_encoder'):
                # For policy types with obs_encoder
                if fast_policy.obs_as_cond:
                    # Handle observation as condition case
                    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                    nobs_features = fast_policy.obs_encoder(this_nobs)
                    # Reshape back to B, To, Do
                    cond = nobs_features.reshape(B, To, -1)
                    # Create action sample
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                else:
                    # Handle observation through repaint condition case
                    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                    nobs_features = fast_policy.obs_encoder(this_nobs)
                    # Reshape back to B, To, Do
                    nobs_features = nobs_features.reshape(B, To, -1)
                    # Create sample containing action and observation features
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                    # Set observation feature part
                    sample[:,:To,action_dim:] = nobs_features
            else:
                # For policies like DiffusionTransformerLowdimPolicy without obs_encoder
                if hasattr(fast_policy, 'obs_as_cond') and fast_policy.obs_as_cond:
                    # If observation as condition
                    cond = obs_dict['obs'][:,:To]
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                    if hasattr(fast_policy, 'pred_action_steps_only') and fast_policy.pred_action_steps_only:
                        # If only predicting action steps
                        sample = torch.zeros(size=(B, fast_policy.n_action_steps, action_dim), device=device, dtype=dtype)
                else:
                    # Observation handled through repaint condition
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                    sample[:,:To,action_dim:] = obs_dict['obs'][:,:To]
            
            try:
                # Use thop.profile to compute FLOPs
                fast_policy.eval()  # Ensure model is in evaluation mode
                macs, params = profile(fast_policy.model, inputs=(sample, timestep, cond), verbose=False)
                
                # MACs (multiply-accumulate operations) are typically considered twice FLOPs
                flops_value = macs * 2
                
                logger.info(f"Model parameter count: {params/1e6:.2f} M")
                logger.info(f"Model MACs: {macs/1e9:.4f} G")
                logger.info(f"Model FLOPs: {flops_value/1e9:.4f} G")
            except Exception as e:
                logger.error(f"Error computing FLOPs: {e}")
                flops_value = 0.0
                import traceback
                traceback.print_exc()
    except Exception as e:
        logger.error(f"Error computing FLOPs: {e}")
        flops_value = 0.0
        import traceback
        traceback.print_exc()

    
    # Performance testing
    logger.info(f"\n=== Performance Testing (action frames/inference: {actions_per_inference}) ===")
    
    # Fixed number of test trials
    num_trials = 10
    
    # Test original policy
    logger.info("\n----- Original Policy -----")
    original_durations = []
    
    for i in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            original_policy.predict_action(obs_dict)
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        original_durations.append(duration)
        logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")
    
    # Calculate original policy statistics
    avg_original = np.mean(original_durations)
    frequency_original = actions_per_inference / avg_original
    logger.info(f"Average time: {avg_original:.4f} seconds")
    logger.info(f"Action frequency: {frequency_original:.2f} actions/second")
    
    # Test cached policy
    logger.info("\n----- Cached Policy -----")
    fast_durations = []
    
    for i in range(num_trials):
        #fast_policy.reset_cache()
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            fast_policy.predict_action(obs_dict)
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        fast_durations.append(duration)
        logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")
    
    # Calculate cached policy statistics
    avg_fast = np.mean(fast_durations)
    frequency_fast = actions_per_inference / avg_fast
    speedup_time = avg_original / avg_fast
    
    logger.info(f"Average time: {avg_fast:.4f} seconds")
    logger.info(f"Action frequency: {frequency_fast:.2f} actions/second")
    logger.info(f"Speedup: {speedup_time:.2f}x")
    
    # Save results
    cache_config = {
        "device": str(device),  # Convert device to string
        "mode": cache_mode,
        "actions_per_inference": int(actions_per_inference),
        "num_bu_blocks": int(num_bu_blocks),
    }
    
    
    if cache_mode == 'optimal':
        cache_config.update({
            "optimal_steps_dir": optimal_steps_dir,
            "metric": metric,
            "num_caches": num_caches
        })
    else:  # threshold
        cache_config.update({
            "cache_threshold": cache_threshold,
        })
    
    # Save benchmark results
    benchmark_results = {
        **cache_config,
        "original": {
            "avg_time": float(avg_original),
            "frequency": float(frequency_original),
        },
        "fast": {
            "avg_time": float(avg_fast),
            "frequency": float(frequency_fast),
        },
        "speedup": float(speedup_time),
        "flops": float(flops_value),
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_dir}/benchmark_results.json")
    
    # Create environment runner
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    if skip_video:
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        logger.info("Skip video rendering")
    
    # Create environment runner
    try:
        env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    except Exception as e:
        logger.error(f"Error instantiating environment runner: {e}")
        try:
            # If failed, it may be due to parameter mismatch, try checking required parameters for KitchenLowdimRunner initialization
            import inspect
            from diffusion_policy.env_runner.kitchen_lowdim_runner import KitchenLowdimRunner
            from diffusion_policy.env_runner.block_push_lowdim_runner import BlockPushLowdimRunner
            
            runner_class = None
            if "KitchenLowdimRunner" in str(env_runner_cfg):
                runner_class = KitchenLowdimRunner
                logger.info("Detected KitchenLowdimRunner")
            elif "BlockPushLowdimRunner" in str(env_runner_cfg):
                runner_class = BlockPushLowdimRunner
                logger.info("Detected BlockPushLowdimRunner")
            
            if runner_class:
                sig = inspect.signature(runner_class.__init__)
                logger.info(f"Runner initialization parameters: {list(sig.parameters.keys())}")
                
                # Adjust env_runner_cfg to match parameters
                required_params = set(sig.parameters.keys()) - {'self'}
                for param in list(env_runner_cfg.keys()):
                    if param not in required_params and param != '_target_':
                        logger.info(f"Remove unneeded parameter: {param}")
                        del env_runner_cfg[param]
                
                # Ensure output_dir is included in parameters
                env_runner_cfg['output_dir'] = output_dir
                
                # Retry instantiation
                env_runner = hydra.utils.instantiate(env_runner_cfg)
            else:
                raise e
        except Exception as inner_e:
            logger.error(f"Attempt to fix environment runner instantiation failed: {inner_e}")
            raise
    
    # Evaluate using cached policy
    logger.info("\nRunning environment evaluation with cached policy...")
    #fast_policy.reset_cache()
    fast_runner_log = env_runner.run(fast_policy)
    
    # Only keep test_mean_score
    test_mean_score = fast_runner_log.get('test/mean_score', 0.0)
    
    # Save evaluation results
    eval_results = {
        "mean_score": float(test_mean_score),
        "speedup": float(speedup_time),
        "flops": float(flops_value),
        "cache_mode": cache_mode,
        "num_caches": num_caches if cache_mode in ['optimal'] else None,
        "num_bu_blocks": int(num_bu_blocks)
    }
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}/eval_results.json")
    logger.info(f"Test success rate: {test_mean_score:.4f}, Speedup: {speedup_time:.2f}x")
    if flops_value > 0:
        logger.info(f"FLOPs: {flops_value/1e9:.4f} GFLOPs")
    
    # Save detailed logs
    json_log = {k: v._path if hasattr(v, '_path') else v for k, v in fast_runner_log.items()}
    out_path = os.path.join(output_dir, 'fast_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f"Detailed evaluation logs saved to {out_path}")

if __name__ == "__main__":
    main()
