# 导入核心功能模块
from HASInfer.analysis.collect_activations import collect_activations
from HASInfer.analysis.activation_analysis import compute_similarity_matrix, load_activations
from HASInfer.analysis.get_optimal_cache_update_steps import get_optimal_cache_update_steps
from HASInfer.analysis.bu_block_selection import (
    analyze_block_errors, 
    select_top_error_blocks, 
    compute_block_l1_errors,
    get_analysis_output_dir
)
from HASInfer.analysis.compute_derivatives import compute_derivatives_for_activations

__all__ = [
    'collect_activations',
    'compute_similarity_matrix',
    'load_activations',
    'get_optimal_cache_update_steps',
    'analyze_block_errors',
    'select_top_error_blocks',
    'compute_block_l1_errors',
    'get_analysis_output_dir',
    'compute_derivatives_for_activations'
] 