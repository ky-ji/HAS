"""
BACInfer - Block-wise Adaptive Caching Inference Module

A plugin-style implementation for accelerating diffusion policy inference
using block-wise adaptive caching techniques.

Components:
- core: Core acceleration module (FastDiffusionPolicy)
- analysis: Activation analysis and optimization tools
- scripts: Evaluation and data collection scripts
- shell: Shell scripts for batch processing

Usage:
    # Import core acceleration
    from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
    
    # Apply BAC acceleration to policy
    policy = FastDiffusionPolicy.apply_cache(
        policy=policy,
        cache_mode='optimal',
        optimal_steps_dir='path/to/optimal_steps',
        num_caches=10,
        metric='cosine',
        num_bu_blocks=3
    )
"""

__version__ = "1.0.0"

# Import core components
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy

# Import analysis tools
from BACInfer.analysis import (
    collect_activations,
    compute_similarity_matrix,
    load_activations,
    get_optimal_cache_update_steps,
    analyze_block_errors,
)

__all__ = [
    'FastDiffusionPolicy',
    'collect_activations',
    'compute_similarity_matrix',
    'load_activations',
    'get_optimal_cache_update_steps',
    'analyze_block_errors',
]

