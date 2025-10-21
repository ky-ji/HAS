#!/usr/bin/env python3

"""
Unified activation collection tool supporting multiple cache modes.

Usage examples:
# Collect activations from original model
python -m diffusion_policy.activation_utils.collect_activations -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization -t square_ph -d cuda:0 --demo_idx 10007 --cache_mode original

# Collect activations from threshold cache model
python -m diffusion_policy.activation_utils.collect_activations -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization -t square_ph -d cuda:0 --demo_idx 10007 --cache_mode threshold --cache_threshold 20 --force

# Collect activations from optimal step cache model
python -m diffusion_policy.activation_utils.collect_activations -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization -t square_ph -d cuda:0 --demo_idx 10007 --cache_mode optimal --optimal_steps_dir assets/square_ph/original/optimal_steps/cosine --metric cosine --num_caches 5 --bu --force

# Collect activations from fix cache model
python -m diffusion_policy.activation_utils.collect_activations -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization -t square_ph -d cuda:0 --demo_idx 10007 --cache_mode fix --optimal_steps_dir assets/square_ph/original/optimal_steps/cosine --metric cosine --num_caches 5 --bu --force

# Collect activations from propagate cache model
python -m diffusion_policy.activation_utils.collect_activations -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization -t square_ph -d cuda:0 --demo_idx 10007 --cache_mode propagate --interpolation_ratio 0.2 --force --reference_activations_path visualization/square_ph/original/activations.pkl


# Collect activations from edit cache model with custom steps
python -m diffusion_policy.activation_utils.collect_activations -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o visualization -t square_ph -d cuda:0 --demo_idx 10007 --cache_mode edit --edit_steps 0,14,31,47,64 --force
"""
import sys
import os
# Get the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory
root_dir = os.path.dirname(os.path.dirname(current_dir))
# Add to Python path
sys.path.append(root_dir)

import torch
import logging
from typing import Dict, List, Set
import pickle
from pathlib import Path
import torch.nn as nn
import copy
import random
import numpy as np
import torch.backends.cudnn as cudnn

# 设置环境变量以启用确定性算法
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 全局默认随机种子
DEFAULT_RANDOM_SEED = 11

# 全局函数设置随机种子
def set_global_seed(seed=DEFAULT_RANDOM_SEED):
    """设置所有随机数生成器的种子"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner
        cudnn.deterministic = True

# Import unified policy runner interface
from BACInfer.analysis.run_policy import run_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivationCollector:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.current_timestep = -1
        self.tracked_modules = set()
        self.modules_seen_this_step = set()
        self.last_activations = {}
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            # Record that this module was seen in the current timestep
            self.modules_seen_this_step.add(name)
            
            if isinstance(output, torch.Tensor):
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(output.detach().cpu())
                # Store the latest activation for this module
                self.last_activations[name] = output.detach().cpu()
            # Handle multi-head attention outputs
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(output[0].detach().cpu())
                # Store the latest activation for this module
                self.last_activations[name] = output[0].detach().cpu()
        return hook
    
    def register_hooks(self, model):
        """Register hooks to all target modules, including parameter modules and dropout modules"""
        logger.info("Starting to scan model structure...")
        
        # Register hooks for all modules
        hook_count = 0
        dropout_count = 0
        param_count = 0
        
        # Define dropout module names to exclude
        #exclude_dropout_names = ["drop"]
        exclude_dropout_names = []
        
        for name, module in model.named_modules():
            should_register = False
            
            # Check if it's a module with parameters
            if len(list(module.parameters())) > 0:
                should_register = True
                param_count += 1
            
            # Check if it's a dropout module, excluding specific names
            elif isinstance(module, nn.Dropout) and name not in exclude_dropout_names:
                should_register = True
                dropout_count += 1
                logger.info(f"Found dropout module: {name}")
            
            # Register hook
            if should_register:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                hook_count += 1
                self.tracked_modules.add(name)
        
        logger.info(f"Registered {hook_count} hooks (parameter modules: {param_count}, dropout modules: {dropout_count})")
    
    def set_timestep(self, timestep):
        if self.current_timestep != -1 and timestep != self.current_timestep:
            self.handle_step_completion()
        
        self.current_timestep = timestep
        self.modules_seen_this_step = set()
    
    def handle_step_completion(self):
        """Check for missing activations at the end of a timestep and fill them with previous values"""
        missing_modules = self.tracked_modules - self.modules_seen_this_step
        
        if missing_modules:
            logger.debug(f"Timestep {self.current_timestep}: Missing activations for {len(missing_modules)} modules")
            
            for module_name in missing_modules:
                # Only fill in missing activations if we have previous values
                if module_name in self.last_activations:
                    if module_name not in self.activations:
                        self.activations[module_name] = []
                    
                    # Append the latest activation we have for this module
                    self.activations[module_name].append(self.last_activations[module_name])
                    logger.debug(f"  - Reused previous activation for {module_name}")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_activations(self):
        """Return collected activations"""
        return self.activations
    
    def save_activations(self, save_path):
        """Save activations to file"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.activations, f)
        logger.info(f"Saved activations for {len(self.activations)} modules to {save_path}")

def collect_activations(
    checkpoint, 
    output_base_dir, 
    task_name,
    device='cuda:0', 
    demo_idx=0, 
    force_recompute=False,
    cache_mode='original', 
    cache_threshold=5, 
    optimal_steps_dir=None, 
    num_caches=30, 
    metric='cosine',
    num_bu_blocks=3,
    edit_steps=None,
    interpolation_ratio=1.0,
    reference_activations_path=None,
    random_seed=DEFAULT_RANDOM_SEED
):
    """
    Collect model activations during a single prediction process, supporting different cache modes
    
    Args:
        checkpoint: Model checkpoint path
        output_base_dir: Base output directory
        task_name: Task name for output path construction
        device: Device to run on
        demo_idx: Demo index
        force_recompute: Whether to force recompute even if activation file exists
        cache_mode: Cache mode ('original', 'threshold', 'optimal', 'fix', 'propagate', 'edit')
        cache_threshold: Cache threshold
        optimal_steps_dir: Path to optimal steps directory
        num_caches: Number of cache updates
        metric: Similarity metric type
        num_bu_blocks: Number of blocks to apply the BU algorithm, 0 to disable
        edit_steps: Custom steps list for edit mode (comma-separated in CLI)
        interpolation_ratio: Interpolation ratio between cached and original activations (default=1.0)
        reference_activations_path: Path to reference (original) activations for interpolation
        random_seed: Random seed for reproducibility (default=11)
        
    Returns:
        Dictionary of collected activations
    """
    # 重新设置随机种子，确保每次调用时都是确定性的
    if random_seed is not None:
        logger.info(f"Setting random seed to {random_seed}")
        set_global_seed(random_seed)
    
    # Build unified output directory structure
    # Basic format: {output_base_dir}/{task_name}/{cache_mode}_{params}/
    output_base_dir = Path(output_base_dir)
    
    # Construct subdirectory name based on cache mode
    if cache_mode == 'original':
        cache_dir_name = 'original'
    elif cache_mode == 'threshold':
        cache_dir_name = f'threshold_{cache_threshold}'
    elif cache_mode == 'optimal':
        cache_dir_name = f'optimal_{metric}_caches_{num_caches}'
    elif cache_mode == 'fix':
        cache_dir_name = f'fix_{metric}_caches_{num_caches}'
        # 确保fix模式必须有optimal_steps_dir
        if optimal_steps_dir is None:
            logger.error("Fix mode requires optimal_steps_dir to be specified")
            raise ValueError("Fix mode requires optimal_steps_dir to be specified")
    elif cache_mode == 'propagate':
        cache_dir_name = 'propagate_mode'
    elif cache_mode == 'edit':
        steps_str = '_'.join(map(str, edit_steps)) if edit_steps else 'default'
        cache_dir_name = f'edit_steps_{steps_str}'
    else:
        cache_dir_name = 'error!!'
        assert(f"Unsupported cache mode: {cache_mode}")
    
    # Add interpolation ratio information to directory name if using interpolation
    if interpolation_ratio < 1.0 and reference_activations_path:
        cache_dir_name = f"{cache_dir_name}_interp_{interpolation_ratio:.2f}"
    
    # Construct full output path
    output_dir = output_base_dir / task_name / cache_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Activation file path
    save_path = output_dir / 'activations.pkl'
    
    # Check if existing activation file exists
    if save_path.exists() and not force_recompute:
        logger.info(f"Found existing activation file: {save_path}")
        try:
            with open(save_path, 'rb') as f:
                activations = pickle.load(f)
            logger.info("Successfully loaded existing activations")
            return activations
        except Exception as e:
            logger.warning(f"Error loading existing file: {e}")
            logger.info("Will recompute activations")
    
    # If we need to recompute activations
    logger.info(f"Computing activations using {cache_mode} mode...")
    logger.info(f"Cache parameters: threshold={cache_threshold}, num_caches={num_caches}, metric={metric}")
    if optimal_steps_dir:
        logger.info(f"  optimal_steps_dir={optimal_steps_dir}")
    if edit_steps:
        logger.info(f"  edit_steps={edit_steps}")
    if num_bu_blocks > 0:
        logger.info(f"  Using BU algorithm with {num_bu_blocks} blocks")
    if interpolation_ratio < 1.0:
        logger.info(f"  interpolation_ratio={interpolation_ratio}")
        if reference_activations_path:
            logger.info(f"  reference_activations_path={reference_activations_path}")
    
    # Use unified policy runner interface to get model and input data
    policy, obs_dict = run_policy(
        checkpoint=checkpoint,
        output_dir=str(output_dir),
        device=device,
        demo_idx=demo_idx,
        cache_mode=cache_mode,
        cache_threshold=cache_threshold,
        optimal_steps_dir=optimal_steps_dir,
        num_caches=num_caches,
        metric=metric,
        num_bu_blocks=num_bu_blocks,
        edit_steps=edit_steps,
        interpolation_ratio=interpolation_ratio,
        reference_activations_path=reference_activations_path,
        random_seed=random_seed
    )
    
    # 验证缓存模式是否正确应用
    if cache_mode != 'original':
        if not hasattr(policy, '_cache'):
            logger.error(f"ERROR: 缓存模式 {cache_mode} 未被正确应用！模型上没有找到 _cache 属性")
            raise RuntimeError(f"Cache mode {cache_mode} was not properly applied to the model")
        
        actual_mode = policy._cache.get('mode')
        if actual_mode != cache_mode:
            logger.error(f"ERROR: 请求的缓存模式为 {cache_mode}，但实际应用的是 {actual_mode}")
            raise RuntimeError(f"Requested cache mode {cache_mode} but actual mode is {actual_mode}")
            
        logger.info(f"已验证缓存模式 {cache_mode} 已正确应用")
        
        # 如果是 fix 或 optimal 模式，验证是否正确加载了步骤
        if cache_mode in ['fix', 'optimal', 'edit']:
            block_steps = policy._cache.get('block_steps', {})
            if not block_steps:
                logger.error(f"错误：{cache_mode} 模式应该有缓存步骤，但是 block_steps 为空！")
                raise RuntimeError(f"Cache mode {cache_mode} should have block_steps but none were loaded")
            logger.info(f"已加载 {len(block_steps)} 个模块的缓存步骤")
            # 打印前几个步骤为示例
            for i, (key, steps) in enumerate(sorted(block_steps.items())):
                if i >= 3:  # 只显示前3个
                    break
                logger.info(f"  {key}: {steps}")
    
    # Set up activation collector
    collector = ActivationCollector()
    
    # Find the correct model to hook
    if hasattr(policy, 'model'):
        # Original model or FastDiffusionPolicy.base_policy.model
        target_model = policy.model
    elif hasattr(policy, 'base_policy') and hasattr(policy.base_policy, 'model'):
        # FastDiffusionPolicy.base_policy.model
        target_model = policy.base_policy.model
    else:
        # If standard model structure not found, use policy itself
        logger.warning("Could not find standard model structure, using entire policy object")
        target_model = policy
    
    # 记录policy的缓存信息
    if hasattr(policy, '_cache'):
        cache_info = {
            'mode': policy._cache.get('mode', None),
            'block_steps': policy._cache.get('block_steps', {}),
            'threshold': policy._cache.get('threshold', None),
            'num_bu_blocks': policy._cache.get('num_bu_blocks', 0),
            'interpolation_ratio': policy._cache.get('interpolation_ratio', 1.0)
        }
        logger.info(f"Policy cache info: {cache_info}")
    
    collector.register_hooks(target_model)
    
    # Patch the model's forward method to track timesteps
    original_model_forward = target_model.forward
    
    def patched_forward(self, sample, timestep, cond=None, **kwargs):
        # Track the current timestep in the collector
        collector.set_timestep(timestep.item() if hasattr(timestep, 'item') else timestep)
        return original_model_forward(sample, timestep, cond, **kwargs)
    
    # Apply the patch
    import types
    target_model.forward = types.MethodType(patched_forward, target_model)
    
    # Run prediction
    with torch.no_grad():
        # If using cache, reset cache first
        if cache_mode != 'original' and hasattr(policy, 'reset_cache'):
            policy.reset_cache()
        action_dict = policy.predict_action(obs_dict)
    
    # Make sure to handle the final timestep
    collector.handle_step_completion()
    
    # Save activations
    collector.save_activations(save_path)
    
    # Remove hooks and restore original forward method
    collector.remove_hooks()
    target_model.forward = original_model_forward
    
    logger.info(f"Activation collection complete, saved to: {save_path}")
    return collector.get_activations()

def get_activations_path(output_base_dir, task_name, cache_mode, **kwargs):
    """
    Get activation file path for specific configuration
    
    Args:
        output_base_dir: Base output directory
        task_name: Task name
        cache_mode: Cache mode
        **kwargs: Other parameters depending on cache mode
    
    Returns:
        Path to activation file
    """
    output_base_dir = Path(output_base_dir)
    
    # Construct subdirectory name based on cache mode
    if cache_mode == 'original':
        cache_dir_name = 'original'
    elif cache_mode == 'threshold':
        cache_dir_name = f'threshold_{kwargs.get("cache_threshold", 5)}'
    elif cache_mode == 'optimal':
        cache_dir_name = f'optimal_{kwargs.get("metric", "cosine")}_caches_{kwargs.get("num_caches", 30)}'
    elif cache_mode == 'fix':
        cache_dir_name = f'fix_{kwargs.get("metric", "cosine")}_caches_{kwargs.get("num_caches", 5)}'
    elif cache_mode == 'propagate':
        cache_dir_name = 'propagate_mode'
    elif cache_mode == 'edit':
        edit_steps = kwargs.get("edit_steps", [])
        steps_str = '_'.join(map(str, edit_steps)) if edit_steps else 'default'
        cache_dir_name = f'edit_steps_{steps_str}'
    else:
        raise ValueError(f"Unsupported cache mode: {cache_mode}")
    
    # Add interpolation ratio information to directory name if using interpolation
    interpolation_ratio = kwargs.get("interpolation_ratio", 1.0)
    if interpolation_ratio < 1.0 and kwargs.get("reference_activations_path"):
        cache_dir_name = f"{cache_dir_name}_interp_{interpolation_ratio:.2f}"
    
    # Return activation file path
    return output_base_dir / task_name / cache_dir_name / 'activations.pkl'

if __name__ == '__main__':
    import click
    
    @click.command()
    @click.option('-c', '--checkpoint', required=True, help='Model checkpoint path')
    @click.option('-o', '--output_base_dir', required=True, help='Base output directory')
    @click.option('-t', '--task_name', required=True, help='Task name')
    @click.option('-d', '--device', default='cuda:0', help='Device to run on')
    @click.option('--demo_idx', default=0, type=int, help='Demo index')
    @click.option('--force', is_flag=True, help='Force recomputation even if file exists')
    @click.option('--cache_mode', default='original', 
                  type=click.Choice(['original', 'threshold', 'optimal', 'fix', 'propagate', 'edit']), 
                  help='Cache mode')
    @click.option('--cache_threshold', default=5, type=int, help='Cache threshold')
    @click.option('--optimal_steps_dir', default=None, help='Path to optimal steps directory')
    @click.option('--num_caches', default=5, type=int, help='Number of cache updates')
    @click.option('--metric', default='cosine', help='Similarity metric type')
    @click.option('--bu', is_flag=True, help='是否为ffblock执行steps补充算法')
    @click.option('--edit_steps', default=None, help='Custom steps for edit mode (comma-separated integers, e.g., "0,20,40,60,80")')
    @click.option('--interpolation_ratio', default=1.0, type=float, help='Interpolation ratio between cached and original activations (default=1.0)')
    @click.option('--reference_activations_path', default=None, help='Path to reference (original) activations for interpolation')
    @click.option('--random_seed', default=DEFAULT_RANDOM_SEED, type=int, help='Random seed for reproducibility')
    def main(checkpoint, output_base_dir, task_name, device, demo_idx, force,
             cache_mode, cache_threshold, optimal_steps_dir, num_caches, metric,
             bu, edit_steps, interpolation_ratio, reference_activations_path, random_seed):
        """Command-line tool to collect model activations"""
        
        # 在main函数开始就设置全局随机种子
        set_global_seed(random_seed)
        
        # 处理edit_steps参数
        processed_edit_steps = None
        if edit_steps is not None:
            try:
                processed_edit_steps = [int(x.strip()) for x in edit_steps.split(',')]
                logger.info(f"Parsed edit steps: {processed_edit_steps}")
            except Exception as e:
                logger.error(f"Error parsing edit_steps '{edit_steps}': {e}")
                logger.error("Format should be comma-separated integers, e.g. '0,20,40,60,80'")
                return
        
        collect_activations(
            checkpoint=checkpoint, 
            output_base_dir=output_base_dir,
            task_name=task_name,
            device=device, 
            demo_idx=demo_idx, 
            force_recompute=force,
            cache_mode=cache_mode,
            cache_threshold=cache_threshold,
            optimal_steps_dir=optimal_steps_dir,
            num_caches=num_caches,
            metric=metric,
            num_bu_blocks=3 if bu else 0,
            edit_steps=processed_edit_steps,
            interpolation_ratio=interpolation_ratio,
            reference_activations_path=reference_activations_path,
            random_seed=random_seed
        )
    
    # 在主程序开始时就设置随机种子
    set_global_seed(DEFAULT_RANDOM_SEED)
    main() 