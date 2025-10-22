#!/usr/bin/env python3

"""
Compute L1 norm errors for each module and select the N blocks with the highest errors to apply BU algorithm.

Usage examples:
# Compute errors and select top 5 blocks to apply BU algorithm
python -m diffusion_policy.activation_utils.bu_block_selection -o assets -t square_ph --cache_mode original --num_blocks 5
"""

import sys
import os
# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root directory
root_dir = os.path.dirname(current_dir)
# Add to Python path
sys.path.append(root_dir)

import torch
import numpy as np
import pickle
from pathlib import Path
import logging
import click
import re
from tqdm import tqdm
from collections import defaultdict

# Import utility functions
from HASInfer.analysis.activation_analysis import load_activations
# Import unified activation path getter
from HASInfer.analysis.collect_activations import get_activations_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_block_l1_errors(original_activations, cached_activations=None):
    """
    Compute L1 norm errors for each module.
    
    Args:
        original_activations: Original model activations
        cached_activations: Cached model activations, if None then use inter-timestep errors within self
        
    Returns:
        Dictionary mapping block names to average L1 errors
    """
    # Filter block types of interest (sa_block, mha_block, ff_block)
    block_pattern = re.compile(r'.*decoder\.layers\.(\d+)\.dropout(\d+).*')
    
    # Collect mapping from block names to activations
    block_activations = {}
    for module_name in original_activations.keys():
        match = block_pattern.match(module_name)
        if match:
            layer_num = int(match.group(1))
            dropout_type = int(match.group(2))
            
            # Map dropout1 to sa_block, dropout2 to mha_block, dropout3 to ff_block
            block_types = ['sa_block', 'mha_block', 'ff_block']
            if 1 <= dropout_type <= 3:
                block_key = f"decoder.layers.{layer_num}_{block_types[dropout_type-1]}"
                block_activations[block_key] = original_activations[module_name]
    
    # Compute average L1 errors for each block
    block_errors = {}
    
    # Compute errors between different timesteps for each block
    for block_key, activations in block_activations.items():
        num_timesteps = len(activations)
        if num_timesteps <= 1:
            logger.warning(f"Block {block_key} has only {num_timesteps} timesteps, skipping")
            continue
        
        total_error = 0.0
        total_comparisons = 0
        
        # If no cached activations provided, compute average errors between different timesteps within self
        if cached_activations is None:
            # Compute L1 errors between all timestep pairs
            for i in range(num_timesteps):
                for j in range(i+1, num_timesteps):
                    act_i = activations[i]
                    act_j = activations[j]
                    
                    # Skip activations with mismatched shapes
                    if act_i.shape != act_j.shape:
                        continue
                    
                    # Compute L1 error
                    error = torch.mean(torch.abs(act_i - act_j)).item()
                    total_error += error
                    total_comparisons += 1
        else:
            # If cached activations provided, compute errors between original and cached activations
            if block_key in cached_activations:
                cached_acts = cached_activations[block_key]
                min_timesteps = min(num_timesteps, len(cached_acts))
                
                for t in range(min_timesteps):
                    orig_act = activations[t]
                    cache_act = cached_acts[t]
                    
                    # Skip activations with mismatched shapes
                    if orig_act.shape != cache_act.shape:
                        continue
                    
                    # Compute L1 error
                    error = torch.mean(torch.abs(orig_act - cache_act)).item()
                    total_error += error
                    total_comparisons += 1
        
        # Compute average error
        if total_comparisons > 0:
            block_errors[block_key] = total_error / total_comparisons
        else:
            logger.warning(f"Block {block_key} has no valid comparison pairs, skipping")
    
    return block_errors

def select_top_error_blocks(block_errors, num_blocks=5):
    """
    Select the N blocks with the highest errors.
    
    Args:
        block_errors: Dictionary mapping block names to errors
        num_blocks: Number of blocks to select
        
    Returns:
        List of (block_name, error) tuples sorted by error
    """
    # Sort by error in descending order
    sorted_errors = sorted(block_errors.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N
    return sorted_errors[:num_blocks]

def save_block_errors(block_errors, output_path):
    """
    Save block error data.
    
    Args:
        block_errors: Dictionary mapping block names to errors
        output_path: Save path
    """
    with open(output_path, 'wb') as f:
        pickle.dump(block_errors, f)
    logger.info(f"Block error data saved to: {output_path}")

def save_selected_blocks(selected_blocks, output_path):
    """
    Save selected blocks.
    
    Args:
        selected_blocks: List of (block_name, error) tuples
        output_path: Save path
    """
    selected_dict = {block: error for block, error in selected_blocks}
    with open(output_path, 'wb') as f:
        pickle.dump(selected_dict, f)
    logger.info(f"Selected blocks saved to: {output_path}")

def analyze_block_errors(activations_path, output_dir, num_blocks=5):
    """
    Analyze block errors and select blocks with highest errors.
    
    Args:
        activations_path: Path to activations file
        output_dir: Output directory
        num_blocks: Number of blocks to select
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load activations
    logger.info(f"Loading activations: {activations_path}")
    activations = load_activations(activations_path)
    
    # Compute block errors
    logger.info("Computing block errors...")
    block_errors = compute_block_l1_errors(activations)
    
    # Save all block errors
    errors_path = output_dir / 'block_l1_errors.pkl'
    save_block_errors(block_errors, errors_path)
    
    # Select blocks with highest errors
    logger.info(f"Selecting top {num_blocks} error blocks...")
    selected_blocks = select_top_error_blocks(block_errors, num_blocks)
    
    # Save selected blocks
    selected_path = output_dir / f'top_{num_blocks}_error_blocks.pkl'
    save_selected_blocks(selected_blocks, selected_path)
    
    # Output selected blocks
    logger.info(f"Top {num_blocks} error blocks:")
    for block, error in selected_blocks:
        logger.info(f"  {block}: {error:.6f}")
    
    return selected_blocks

def get_analysis_output_dir(output_base_dir, task_name, cache_mode, **kwargs):
    """
    Get the output directory path for analysis results.
    
    Args:
        output_base_dir: Base output directory
        task_name: Task name
        cache_mode: Cache mode
        **kwargs: Other parameters based on cache mode
    
    Returns:
        Output directory path for analysis results
    """
    # Get the directory of the activations path
    activations_dir = get_activations_path(output_base_dir, task_name, cache_mode, **kwargs).parent
    # Build analysis results output directory
    return activations_dir / 'bu_block_selection'

@click.command()
@click.option('-o', '--output_base_dir', required=True, help='Base output directory')
@click.option('-t', '--task_name', required=True, help='Task name')
@click.option('--cache_mode', default='original', 
              type=click.Choice(['original', 'threshold', 'optimal', 'random']), 
              help='Cache mode')
@click.option('--cache_threshold', default=5, type=int, help='Cache threshold')
@click.option('--num_caches', default=30, type=int, help='Number of cache updates')
@click.option('--metric', default='cosine', help='Similarity metric type')
@click.option('--num_blocks', default=5, type=int, help='Number of blocks to select')
@click.option('--force', is_flag=True, help='Force recomputation even if results exist')
def main(output_base_dir, task_name, cache_mode, cache_threshold, 
         num_caches, metric, num_blocks, force):
    """Command line tool for computing block errors and selecting blocks with highest errors to apply BU algorithm"""
    
    # Build parameter dictionary for path constructor
    kwargs = {
        'cache_threshold': cache_threshold,
        'num_caches': num_caches,
        'metric': metric
    }
    
    # Get activations file path
    activations_path = get_activations_path(output_base_dir, task_name, cache_mode, **kwargs)
    
    # Check if activations file exists
    if not activations_path.exists():
        logger.error(f"Activations file does not exist: {activations_path}")
        logger.error(f"Please run collect_activations.py first with cache_mode={cache_mode}")
        return
    
    # Get analysis results output directory
    analysis_output_dir = get_analysis_output_dir(output_base_dir, task_name, cache_mode, **kwargs)
    
    # Check if recomputation is needed
    if analysis_output_dir.exists() and not force:
        selected_path = analysis_output_dir / f'top_{num_blocks}_error_blocks.pkl'
        if selected_path.exists():
            logger.info(f"Analysis results already exist: {selected_path}")
            logger.info("Use --force to force recomputation")
            return
    
    # Create output directory
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze block errors
    logger.info(f"Analyzing block errors for {cache_mode} mode")
    analyze_block_errors(str(activations_path), str(analysis_output_dir), num_blocks)
    
    logger.info(f"Analysis completed, results saved to: {analysis_output_dir}")

if __name__ == '__main__':
    main() 