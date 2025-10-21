#!/usr/bin/env python3

# Example usage commands:
# python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py -c checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt -o assets/square_ph -d cuda:2 --num_caches '5,8,10,20' --force_recompute
# python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py -c checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt -o assets/can_ph -d cuda:2 --num_caches '5,8,10,20' --force_recompute
# python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py -c checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt -o assets/tool_hang_ph -d cuda:2 --num_caches '5,8,10,20' --force_recompute
# python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py -c checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt -o assets/lift_ph -d cuda:2 --num_caches '5,8,10,20' --force_recompute

import sys
import os
# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root directory
root_dir = os.path.dirname(os.path.dirname(current_dir))
# Add to Python path
sys.path.append(root_dir)

import logging
import click
import torch
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union


# Import related functional modules
from BACInfer.analysis.run_policy import run_policy
from BACInfer.analysis.collect_activations import collect_activations
from BACInfer.analysis.activation_analysis import (
    compute_similarity_matrix,
    load_activations,
    save_similarity_matrices,
    get_module_metrics_from_similarity_data
)
# Import optimal cache scheduler
from BACInfer.analysis.optimal_cache_scheduler import (
    compute_optimal_cache_steps,
    OptimalCacheScheduler
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimal_cache_update_steps(
    checkpoint: str,
    output_dir: str,
    device: str = 'cuda:0',
    demo_idx: int = 0,
    force_recompute: bool = False,
    metrics: List[str] = None,
    num_caches_list: List[int] = None
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    Compute optimal cache update schedules for dropout modules in decoder.layers.

    Args:
        checkpoint: Path to model checkpoint
        output_dir: Output directory
        device: Device to run on, default 'cuda:0'
        demo_idx: Demo sample index, default 0
        force_recompute: Force recompute all steps, default False
        metrics: List of similarity metrics to compute
        num_caches_list: List of cache counts to compute

    Returns:
        Dictionary containing all computation results, structure: {metric: {module_name: {num_caches: optimal_steps}}}
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Step 1: Collect activations
    activations_path = output_dir /'original'/'activations.pkl'
    if not activations_path.exists() or force_recompute:
        logger.info(f"Collecting activations...")
        # Extract task name
        task_name = output_dir.name
        output_base_dir = str(output_dir.parent)
        collect_activations(
            checkpoint=checkpoint,
            output_base_dir=output_base_dir,
            task_name=task_name,
            device=device,
            demo_idx=demo_idx,
            force_recompute=force_recompute,
            cache_mode='original'
        )
        logger.info(f"Activations saved to {activations_path}")
    else:
        logger.info(f"Found existing activation file: {activations_path}")


    similarity_matrices_path = output_dir /'original'/'similarity_matrices.pkl'

    if not similarity_matrices_path.exists() or force_recompute:
        logger.info(f"Computing similarity matrices...")
        # Create similarity matrix directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load activations
        activations_dict = load_activations(str(activations_path))

        # Only compute similarity matrices for dropout modules in decoder.layers and first layer self_attn
        results = {}
        pattern = re.compile(r'(decoder\.layers\.\d+\.dropout[1-3]|decoder\.layers\.0\.self_attn)')

        for module_name, module_activations in activations_dict.items():
            # Only process dropout modules in decoder.layers and first layer self_attn
            if not pattern.match(module_name):
                logger.info(f"Skipping module: {module_name}, not a target module")
                continue

            logger.info(f"Processing module: {module_name}")

            try:
                # Compute similarity matrix
                sim_matrices = compute_similarity_matrix(module_activations)
                results[module_name] = sim_matrices
            except Exception as e:
                logger.error(f"Error computing similarity matrix for module {module_name}: {str(e)}")
                continue

        # Save similarity matrices
        save_similarity_matrices(results, similarity_matrices_path)
        logger.info(f"Similarity matrices saved to {similarity_matrices_path}")
    else:
        logger.info(f"Found existing similarity matrices file: {similarity_matrices_path}")


    # Get all available modules and metrics from similarity data
    module_metrics = get_module_metrics_from_similarity_data(str(output_dir / 'original'))

    # Filter dropout modules in decoder.layers and first layer self_attn
    pattern = re.compile(r'(decoder\.layers\.\d+\.dropout[1-3]|decoder\.layers\.0\.self_attn)')
    module_names = [m for m in module_metrics.keys() if pattern.match(m)]

    if not module_names:
        logger.warning("No dropout modules in decoder.layers or first layer self_attn found, check activation collection")
        return {}

    logger.info(f"Computing optimal cache steps for modules: {module_names}")

    # Step 3: Compute optimal cache steps
    results = {}

    for metric in metrics:
        if metric not in results:
            results[metric] = {}

        for module_name in module_names:
            # Ensure module supports requested metric
            if module_name in module_metrics and metric in module_metrics[module_name]:
                if module_name not in results[metric]:
                    results[metric][module_name] = {}

                for num_caches in num_caches_list:
                    # Build cache step file path
                    cache_filename = f"optimal_steps_{module_name}_{num_caches}_{metric}.pkl"
                    cache_path = output_dir / 'optimal_steps'/ metric / module_name / cache_filename

                    if not cache_path.exists() or force_recompute:
                        logger.info(f"Computing optimal cache steps: module={module_name}, metric={metric}, num_caches={num_caches}")
                        try:
                            # Compute optimal steps
                            optimal_steps = compute_optimal_cache_steps(
                                similarity_matrices_dir=str(output_dir/'original'),
                                module_name=module_name,
                                num_caches=num_caches,
                                metric=metric
                            )
                            results[metric][module_name][num_caches] = optimal_steps
                            logger.info(f"Optimal cache steps saved to {cache_path}")
                        except Exception as e:
                            logger.error(f"Error computing optimal cache steps: {str(e)}")
                            continue
                    else:
                        logger.info(f"Found existing optimal cache steps file: {cache_path}")
                        # Load existing optimal steps
                        optimal_steps = OptimalCacheScheduler.load_optimal_steps(str(cache_path))
                        if optimal_steps:
                            results[metric][module_name][num_caches] = optimal_steps
                        else:
                            logger.warning(f"Failed to load optimal steps file {cache_path}")
            else:
                logger.warning(f"Module {module_name} does not support metric {metric}")

    return results


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Model checkpoint path')
@click.option('-o', '--output_dir', required=True, help='Output directory')
@click.option('-d', '--device', default='cuda:0', help='Device')
@click.option('--demo_idx', default=0, type=int, help='Demo sample index')
@click.option('--force_recompute', is_flag=True, help='Force recompute all steps')
@click.option('--metrics', default='cosine,l1,mse', help='Similarity metrics to compute, comma-separated, e.g.: cosine,mse,l1')
@click.option('--num_caches', default='5,8,10,20', help='Cache counts to compute, comma-separated, e.g.: 5,8,10,20')

def main(checkpoint, output_dir, device, demo_idx, force_recompute, metrics, num_caches):
    """Compute optimal cache steps for dropout modules in decoder.layers"""
    # Parse metrics string to list
    metrics_list = metrics.split(',') if metrics else None

    # Parse num_caches string to integer list
    num_caches_list = [int(n.strip()) for n in num_caches.split(',')] if num_caches else None

    # Execute integrated analysis
    results = get_optimal_cache_update_steps(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        demo_idx=demo_idx,
        force_recompute=force_recompute,
        metrics=metrics_list,
        num_caches_list=num_caches_list
    )

    # Output analysis results summary
    logger.info("Analysis complete. Results summary:")
    if not results:
        logger.warning("No results generated")
        return

    for metric, module_dict in results.items():
        logger.info(f"Metric: {metric}")
        for module_name, cache_dict in module_dict.items():
            logger.info(f"  Module: {module_name}")
            for num_caches, optimal_steps in cache_dict.items():
                logger.info(f"    Cache count: {num_caches}, steps: {len(optimal_steps)}")
                if len(optimal_steps) <= 10:
                    logger.info(f"      Steps: {optimal_steps}")
                else:
                    logger.info(f"      First 10 steps: {optimal_steps[:10]}...")

                # Display viewable file paths
                cache_filename = f"optimal_steps_{module_name}_{num_caches}_{metric}"
                txt_path = os.path.join(output_dir, 'optimal_steps', metric, module_name, f"{cache_filename}.txt")
                json_path = os.path.join(output_dir, 'optimal_steps', metric, module_name, f"{cache_filename}.json")

                if os.path.exists(txt_path):
                    logger.info(f"      View TXT file: {txt_path}")
                if os.path.exists(json_path):
                    logger.info(f"      View JSON file: {json_path}")


if __name__ == '__main__':
    main()