"""
HASInfer - Hash Inference Module

A plugin-style implementation for accelerating diffusion policy inference
using block-wise adaptive caching techniques.

Components:
- core: Core acceleration module (FastDiffusionPolicy)
- analysis: Activation analysis and optimization tools
- scripts: Evaluation and data collection scripts

Usage:
    # Import core acceleration
    from HASInfer.core.diffusion_hash_wrapper_multistep import FastDiffusionPolicyMultistep
    
    # Apply HAS acceleration to policy
    fast_policy = FastDiffusionPolicyMultistep.apply_cache(
        policy=fast_policy,
        cache_mode='optimal',
        optimal_steps_dir=optimal_steps_dir,
        num_caches=num_caches,
        metric=metric,
        num_bu_blocks=num_bu_blocks,
        precomputed_derivatives_path=precomputed_derivatives_path,
        multistep_method=multistep_method,
        max_queue_length=max_queue_length
    )
"""

__version__ = "1.0.0"

# Import core components
from HASInfer.core.diffusion_hash_wrapper_multistep import FastDiffusionPolicyMultistep

# Import analysis tools
from HASInfer.analysis import (
    collect_activations,
    compute_similarity_matrix,
    load_activations,
    get_optimal_cache_update_steps,
    analyze_block_errors,
    compute_derivatives_for_activations,
)

__all__ = [
    'FastDiffusionPolicy',
    'collect_activations',
    'compute_similarity_matrix',
    'load_activations',
    'get_optimal_cache_update_steps',
    'analyze_block_errors',
    'compute_derivatives_for_activations',
]

