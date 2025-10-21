import torch
import types
import logging
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch.nn as nn
import re

# Import necessary classes
# Note: Keep original project imports for TransformerForDiffusion
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

# Create global logger
logger = logging.getLogger(__name__)

class FastDiffusionPolicy:
    """
    BAC (Block-wise Adaptive Caching) wrapper for accelerating diffusion policies.

    Supports three caching modes:
    1. original: No caching, baseline inference
    2. threshold: Cache updates at fixed intervals
    3. optimal: Adaptive per-block caching schedules (ACS + BU algorithm)

    Usage:
        policy = workspace.get_policy()  # Get original policy

        # No caching (baseline)
        FastDiffusionPolicy.apply_cache(policy, cache_mode='original')

        # Fixed threshold caching
        FastDiffusionPolicy.apply_cache(policy, cache_threshold=5)

        # Optimal adaptive caching with BU algorithm
        FastDiffusionPolicy.apply_cache(
            policy,
            cache_mode='optimal',
            optimal_steps_dir='path/to/steps_dir',
            num_caches=5,
            metric='cosine',
            num_bu_blocks=3
        )
    """

    @staticmethod
    def apply_cache(policy,
                   cache_threshold: int = 5,
                   optimal_steps_dir: str = None,
                   num_caches: int = 30,
                   metric: str = 'cosine',
                   cache_mode: str = None,
                   num_bu_blocks: int = 0):
        """
        Apply BAC caching acceleration to a diffusion policy.

        Args:
            policy: Original diffusion policy with 'model' attribute
            cache_threshold: Fixed interval for cache updates (threshold mode)
            optimal_steps_dir: Directory containing optimal cache schedules (optimal mode)
            num_caches: Number of cache updates per block (for loading optimal steps)
            metric: Similarity metric used for optimal schedules ('cosine', 'l1', 'mse')
            cache_mode: Caching mode - 'original', 'threshold', or 'optimal'
            num_bu_blocks: Number of blocks for BU algorithm (0 disables BU)

        Returns:
            Modified policy instance (in-place modification) or original policy (if cache_mode='original')
        """
        # Determine cache mode
        if cache_mode is None:
            # Auto-infer mode
            if optimal_steps_dir is not None:
                cache_mode = 'optimal'
            else:
                cache_mode = 'threshold'

        # Return original policy if no caching
        if cache_mode == 'original':
            logger.info("Using original mode, no caching applied")
            return policy

        assert hasattr(policy, 'model')
        model = policy.model
        assert isinstance(model, TransformerForDiffusion)
        num_inference_steps = policy.num_inference_steps

        # Create cache structure
        cache = {
            'threshold': cache_threshold,      # Fixed interval for cache updates
            'mode': cache_mode,                # Cache mode: 'original', 'threshold', 'optimal'
            'metric': metric,                  # Similarity metric (cosine, l1, mse)
            'optimal_steps_dir': optimal_steps_dir,  # Directory for optimal schedules
            'num_caches': num_caches,          # Number of cache updates
            'num_steps': num_inference_steps,  # Total denoising steps
            'current_step': -1,                # Current step counter (starts at 0)
            'block_cache': {},                 # Block cache storage: {block_key: cached_output}
            'block_steps': {},                 # Per-block schedules: {block_key: [update_steps]}
            'num_bu_blocks': num_bu_blocks,    # Number of blocks for BU algorithm (0 disables)
        }
        policy._cache = cache

        # Find all cacheable transformer layers
        cacheable_layers = FastDiffusionPolicy._find_cacheable_layers(model)
        policy._cacheable_layers = cacheable_layers
        logger.info(f"Found {len(cacheable_layers)} cacheable Transformer layers")

        # Load optimal schedules for optimal mode
        if cache_mode == 'optimal' and optimal_steps_dir:
            # Load per-block optimal cache update schedules
            FastDiffusionPolicy._load_block_optimal_steps(cache, cacheable_layers, optimal_steps_dir, num_caches, metric)
            cache_threshold = None

        # Add caching functionality to each layer
        for layer_name, layer in cacheable_layers:
            FastDiffusionPolicy._add_cache_to_block(layer, layer_name, cache)

        # Save original forward method
        original_forward = model.forward

        # Create forward method with caching
        def forward_with_cache(self, sample, timestep, cond=None, **kwargs):
            """Forward pass with caching enabled"""
            cache = getattr(policy, '_cache', None)

            # Increment step counter
            cache['current_step'] += 1
            current_step = cache['current_step']

            # Update cache flags for all blocks at current step
            FastDiffusionPolicy._update_cache_flags(cache, current_step)

            # Execute forward pass with caching
            output = original_forward(sample, timestep, cond, **kwargs)

            return output

        # Replace model's forward method
        model.forward = types.MethodType(forward_with_cache, model)

        # Add cache reset method
        def reset_cache(self):
            """Reset cache between inference rollouts"""
            if hasattr(self, '_cache'):
                self._cache['current_step'] = -1
                self._cache['block_cache'] = {}
                logger.debug("Cache reset")
            return self

        # Add method to policy object
        policy.reset_cache = types.MethodType(reset_cache, policy)

        # Save original predict_action method
        original_predict_action = policy.predict_action

        # Create predict_action with auto cache reset
        def predict_action_with_auto_reset(self, *args, **kwargs):
            """Automatically reset cache before each predict_action call"""
            # Reset cache
            self.reset_cache()
            # Call original predict_action
            return original_predict_action(*args, **kwargs)

        # Replace policy's predict_action method
        policy.predict_action = types.MethodType(predict_action_with_auto_reset, policy)

        # Log application info
        if cache_mode == 'threshold':
            logger.info(f"Threshold-based caching applied successfully, threshold: {cache_threshold}")
        elif cache_mode == 'optimal':
            logger.info(f"Optimal ACS caching applied successfully, steps_dir: {optimal_steps_dir}, num_caches: {num_caches}")

        return policy

    @staticmethod
    def _find_cacheable_layers(model) -> List[Tuple[str, Any]]:
        """
        Find all cacheable Transformer layers in the model.

        Args:
            model: TransformerForDiffusion model instance

        Returns:
            List of (layer_name, layer_module) tuples for cacheable layers
        """
        import torch.nn as nn

        # Only search for TransformerDecoderLayer in decoder
        cacheable_layers = []

        # Process decoder layers
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                if isinstance(layer, nn.TransformerDecoderLayer):
                    name = f'decoder.layers.{i}'
                    cacheable_layers.append((name, layer))
                    logger.info(f"Selected decoder layer for caching: {name}")

        # Fallback: search all modules for TransformerDecoderLayer
        if not cacheable_layers:
            for name, module in model.named_modules():
                if isinstance(module, nn.TransformerDecoderLayer):
                    cacheable_layers.append((name, module))
                    logger.info(f"Selected Transformer layer for caching: {name}")

        logger.info(f"Total {len(cacheable_layers)} cacheable Transformer layers found")
        return cacheable_layers

    @staticmethod
    def _load_block_optimal_steps(cache: Dict, layers: List[Tuple[str, Any]],
                                 optimal_steps_dir: str, num_caches: int, metric: str):
        """
        Load optimal cache update schedules for each block in transformer layers.

        For each TransformerDecoderLayer, loads schedules for three computational blocks:
        - sa_block: Self-attention block (uses dropout1 schedules)
        - mha_block: Multi-head cross-attention block (uses dropout2 schedules)
        - ff_block: Feedforward network block (uses dropout3 schedules)

        If num_bu_blocks > 0, applies the Bubbling Union (BU) algorithm to propagate
        cache update steps from deeper to shallower layers, ensuring activation freshness.

        BU Algorithm has two phases:
        1. Each BU block collects steps from all deeper FFN blocks (regardless of BU membership)
        2. Traditional backward propagation within BU blocks (from deeper to shallower)

        This two-phase propagation ensures BU blocks acquire all necessary update steps,
        especially from later-layer FFN blocks.

        Args:
            cache: Cache dictionary
            layers: List of (layer_name, layer_module) tuples
            optimal_steps_dir: Directory containing optimal schedule files
            num_caches: Number of cache updates per block
            metric: Similarity metric (cosine, l1, mse)
        """
        steps_dir = Path(optimal_steps_dir)
        assert steps_dir.exists(), f"Optimal steps directory {optimal_steps_dir} does not exist"

        # Store per-block schedules
        block_steps = {}


        for layer_name, layer in layers:
            # Find corresponding dropout schedule files for three main blocks in this layer
            dropout1_name = f"{layer_name}.dropout1"
            dropout2_name = f"{layer_name}.dropout2"
            dropout3_name = f"{layer_name}.dropout3"

            # sa_block uses dropout1 schedules
            sa_block_key = f"{layer_name}_sa_block"
            steps_file = steps_dir/dropout1_name/f'optimal_steps_{dropout1_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[sa_block_key] = steps
                    logger.info(f"Loaded steps for self-attention block {sa_block_key}: {steps}")
                except Exception as e:
                    logger.warning(f"Failed to load steps for {sa_block_key}: {e}")

            # mha_block uses dropout2 schedules
            mha_block_key = f"{layer_name}_mha_block"
            steps_file = steps_dir/dropout2_name/f'optimal_steps_{dropout2_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[mha_block_key] = steps
                    logger.info(f"Loaded steps for multi-head attention block {mha_block_key}: {steps}")
                except Exception as e:
                    logger.warning(f"Failed to load steps for {mha_block_key}: {e}")

            # ff_block uses dropout3 schedules
            ff_block_key = f"{layer_name}_ff_block"
            steps_file = steps_dir/dropout3_name/f'optimal_steps_{dropout3_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[ff_block_key] = steps
                    logger.info(f"Loaded steps for feedforward block {ff_block_key}: {steps}")
                except Exception as e:
                    logger.warning(f"Failed to load steps for {ff_block_key}: {e}")

        # Apply Bubbling Union (BU) algorithm for FFN blocks if num_bu_blocks > 0
        if cache.get('num_bu_blocks', 0) > 0:
            logger.info("Applying BU (Bubbling Union) algorithm for step propagation")

            # Load or compute BU blocks (blocks with highest caching error)
            bu_blocks = cache.get('bu_blocks', None)

            if bu_blocks is None:
                # Try to load error analysis results to identify high-error blocks
                try:
                    from BACInfer.analysis.bu_block_selection import (
                        analyze_block_errors,
                    )

                    # Derive analysis output directory from optimal_steps_dir
                    # Format example: assets/tool_hang_ph/original/optimal_steps/cosine
                    if optimal_steps_dir:
                        steps_path = Path(optimal_steps_dir)
                        # Extract to original directory level: assets/tool_hang_ph/original
                        if "optimal_steps" in str(steps_path):
                            base_path = steps_path.parent.parent
                            output_base_dir = str(base_path.parent)  # assets
                            task_name = base_path.name.split('/')[0]  # task name
                            logger.info(f"Derived base directory from optimal_steps_dir: output_base_dir={output_base_dir}, task={task_name}")
                        else:
                            logger.info("Cannot derive base directory from optimal_steps_dir")
                    else:
                        logger.info("optimal_steps_dir not provided")

                    num_blocks = cache.get('num_bu_blocks', 3)

                    # Construct BU block selection path
                    bu_analysis_dir = f"{output_base_dir}/{task_name}/bu_block_selection"
                    analysis_output_dir = Path(bu_analysis_dir)
                    logger.info(f"BU analysis output directory: {analysis_output_dir}")

                    selected_path = analysis_output_dir / f'top_{num_blocks}_error_blocks.pkl'

                    if selected_path.exists():
                        # Load selected blocks from file
                        with open(selected_path, 'rb') as f:
                            selected_blocks_dict = pickle.load(f)
                            bu_blocks = list(selected_blocks_dict.keys())
                            logger.info(f"Loaded {len(bu_blocks)} high-error blocks for BU: {bu_blocks}")
                    else:
                        # If file doesn't exist, compute from activations
                        activations_path = Path(f"{output_base_dir}/{task_name}/activations.pkl")
                        if activations_path.exists():
                            logger.info("Computing high-error blocks for BU algorithm...")
                            analysis_output_dir.mkdir(parents=True, exist_ok=True)
                            selected_blocks = analyze_block_errors(str(activations_path), str(analysis_output_dir), num_blocks)
                            bu_blocks = [block for block, _ in selected_blocks]
                            logger.info(f"Computed {len(bu_blocks)} high-error blocks for BU: {bu_blocks}")
                        else:
                            # Cannot compute, skip BU
                            logger.warning(f"Activation file not found: {activations_path}, cannot apply BU")
                            bu_blocks = []
                except Exception as e:
                    logger.warning(f"Error loading/computing high-error blocks: {e}")
                    logger.warning(f"Cannot apply BU algorithm")
                    bu_blocks = []


            # Unified BU propagation (from deeper/later to shallower/earlier)
            if len(bu_blocks) > 1:
                # Sort blocks by layer index to ensure proper order
                # Parse layer index
                def get_layer_idx(block_key):
                    match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
                    if match:
                        return int(match.group(1))
                    return -1

                # Parse block type priority (sa_block=0, mha_block=1, ff_block=2)
                def get_block_type_priority(block_key):
                    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
                    if match:
                        block_type = match.group(1)
                        priorities = {'sa_block': 0, 'mha_block': 1, 'ff_block': 2}
                        return priorities.get(block_type, 10)
                    return 10

                # Parse block type
                def get_block_type(block_key):
                    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
                    if match:
                        return match.group(1)
                    return ""

                # Sort by layer index and block type
                sorted_bu_blocks = sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x)))
                logger.info(f"BU: Sorted blocks: {sorted_bu_blocks}")

                # Phase 1: Collect steps from all deeper FFN blocks
                # Find all FFN blocks (regardless of BU membership) and sort by layer index
                all_ffn_blocks = []
                for block_key in block_steps.keys():
                    if get_block_type(block_key) == 'ff_block':
                        all_ffn_blocks.append(block_key)

                sorted_all_ffn_blocks = sorted(all_ffn_blocks, key=lambda x: get_layer_idx(x))
                logger.info(f"Phase 1, BU: All FFN blocks (sorted by layer): {sorted_all_ffn_blocks}")

                # For each BU block, find all deeper FFN blocks
                for i, block_key in enumerate(sorted_bu_blocks):
                    block_layer_idx = get_layer_idx(block_key)

                    # Collect all deeper FFN blocks
                    deeper_ffn_blocks = []
                    for ffn_block in sorted_all_ffn_blocks:
                        ffn_layer_idx = get_layer_idx(ffn_block)
                        if ffn_layer_idx >= block_layer_idx:
                            deeper_ffn_blocks.append(ffn_block)

                    if deeper_ffn_blocks:
                        logger.info(f"BU: For block {block_key} (layer {block_layer_idx}), found deeper FFN blocks: {deeper_ffn_blocks}")

                        # Merge steps from all deeper FFN blocks
                        all_deeper_steps = set()
                        for deeper_ffn_block in deeper_ffn_blocks:
                            if deeper_ffn_block in block_steps:
                                all_deeper_steps.update(block_steps[deeper_ffn_block])

                        # Ensure current block has all deeper FFN steps
                        if block_key in block_steps and all_deeper_steps:
                            current_steps = set(block_steps[block_key])
                            missing_steps = all_deeper_steps - current_steps

                            if missing_steps:
                                updated_steps = sorted(list(current_steps.union(missing_steps)))
                                block_steps[block_key] = updated_steps
                                logger.info(f"BU: Added missing steps to {block_key}: {sorted(list(missing_steps))}. New steps: {updated_steps}")

                # Phase 2 (optional): Traditional backward propagation within BU blocks
                # Currently commented out - can be enabled if needed
            else:
                logger.info("BU: Insufficient blocks for propagation (need at least 2)")

            # Print BU algorithm summary
            logger.info("\n===== BU Algorithm Summary =====")
            logger.info(f"Total BU blocks processed: {len(bu_blocks)}")

            # Statistics for each block's steps
            for block_key in sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x))):
                if block_key in block_steps:
                    steps_count = len(block_steps[block_key])
                    logger.info(f"  Block {block_key}: {steps_count} steps")


        # Store to cache
        cache['block_steps'] = block_steps

        # Log output
        logger.info(f"Loaded optimal steps for {len(block_steps)} computational blocks")

    @staticmethod
    def _update_cache_flags(cache: Dict, current_step: int):
        """
        Update cache flags for all blocks

        Args:
            cache: Cache dictionary
            current_step: Current step
        """
        cache_mode = cache['mode']

        if cache_mode == 'threshold':
            # Threshold-based caching strategy
            threshold = cache['threshold']

            # Update cache every threshold steps
            # Or force update at steps 99-100 (maintain original behavior)
            should_cache = (current_step % threshold == 0) or (current_step >= 99 and current_step <= 100)
            should_cache = (current_step % threshold == 0)

            cache['should_cache'] = should_cache

        else:
            # Optimal steps, Fix mode, Propagate mode, or Edit mode caching strategy
            # Performance optimization: Pre-store all steps in a dictionary for O(1) lookup
            if 'steps_lookup' not in cache:
                # Create lookup table on first run
                steps_lookup = {}
                for block_key, steps in cache['block_steps'].items():
                    # Empty steps list means compute at all steps (no caching)
                    if not steps:
                        steps_lookup[block_key] = set()
                    else:
                        steps_lookup[block_key] = set(steps)
                        # Add steps 99-100 as forced cache steps (maintain original behavior)
                        steps_lookup[block_key].add(99)
                        steps_lookup[block_key].add(100)
                cache['steps_lookup'] = steps_lookup

            # Use lookup table for O(1) lookup
            should_cache = {}
            for block_key, step_set in cache['steps_lookup'].items():
                # Empty step set means compute at all steps (no caching)
                if not step_set:
                    should_cache[block_key] = False
                else:
                    should_cache[block_key] = current_step in step_set

            cache['should_cache'] = should_cache

    @staticmethod
    def _add_cache_to_block(layer, layer_name: str, cache: Dict):
        """
        Add caching functionality to Transformer layer, directly caching results of self-attention, multi-head attention, and feedforward network

        Args:
            layer: TransformerDecoderLayer
            layer_name: Layer name
            cache: Cache dictionary
        """
        # Save original forward method
        original_forward = layer.forward

        # Remove interpolation related logic

        # Only handle TransformerDecoderLayer, assuming norm_first=True
        if isinstance(layer, nn.TransformerDecoderLayer):
            # Create caching method for TransformerDecoderLayer
            def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                  tgt_is_causal=None, memory_is_causal=False):
                cache_mode = cache['mode']
                # Get global cache_ff_block setting

                cache_ff_block = cache.get('cache_ff_block', True)

                # When layer_name contains "1" or "3" or "4" or "5" or "7", set cache_ff_block to False
                # if "0" in layer_name or "1" in layer_name or "2" in layer_name or "3" in layer_name or "6" in layer_name or "7" in layer_name:
                #     cache_ff_block = False
                # else:
                #     cache_ff_block = True

                # Cache keys
                sa_block_key = f"{layer_name}_sa_block"
                mha_block_key = f"{layer_name}_mha_block"
                ff_block_key = f"{layer_name}_ff_block"

                # Determine whether to update cache
                if cache_mode == 'threshold':
                    should_cache = cache.get('should_cache', False)
                    should_cache_sa = should_cache
                    should_cache_mha = should_cache
                    should_cache_ff = should_cache and cache_ff_block
                else:
                    block_should_cache = cache.get('should_cache', {})
                    should_cache_sa = block_should_cache.get(sa_block_key, False)
                    should_cache_mha = block_should_cache.get(mha_block_key, False)
                    should_cache_ff = block_should_cache.get(ff_block_key, False) and cache_ff_block

                # Check cache for self-attention, multi-head attention, and feedforward network
                has_sa_cache = sa_block_key in cache['block_cache'] and not should_cache_sa
                has_mha_cache = mha_block_key in cache['block_cache'] and not should_cache_mha
                has_ff_cache = ff_block_key in cache['block_cache'] and not should_cache_ff

                # Get current timestep (not directly used)

                # Assuming norm_first=True
                x = tgt

                # Self-attention part
                if has_sa_cache:
                    # Directly use cache
                    x = x + cache['block_cache'][sa_block_key]
                else:
                    # Compute and cache self-attention block
                    sa_result = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    if should_cache_sa:
                        cache['block_cache'][sa_block_key] = sa_result.detach()
                    x = x + sa_result

                # Multi-head attention part
                if has_mha_cache:
                    # Directly use cache
                    x = x + cache['block_cache'][mha_block_key]
                else:
                    # Compute and cache multi-head attention block
                    mha_result = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    if should_cache_mha:
                        cache['block_cache'][mha_block_key] = mha_result.detach()
                    x = x + mha_result

                # Feedforward network part
                if has_ff_cache:
                    # Directly use cache
                    x = x + cache['block_cache'][ff_block_key]
                else:
                    # Compute and cache feedforward network block
                    ff_result = self._ff_block(self.norm3(x))
                    if should_cache_ff:
                        cache['block_cache'][ff_block_key] = ff_result.detach()
                    x = x + ff_result

                return x
        else:
            # For other module types, use default caching method
            def forward_with_cache(self, *args, **kwargs):
                cache_mode = cache['mode']
                # current_step not used
                block_key = layer_name  # Use layer name as block key

                if cache_mode == 'threshold':
                    should_cache = cache.get('should_cache', False)
                else:
                    block_should_cache = cache.get('should_cache', {})
                    should_cache = block_should_cache.get(block_key, False)

                if block_key in cache['block_cache'] and not should_cache:
                    return cache['block_cache'][block_key]

                output = original_forward(*args, **kwargs)

                if should_cache:
                    if isinstance(output, torch.Tensor):
                        cache['block_cache'][block_key] = output.detach()

                return output

        # Replace layer's forward method
        layer.forward = types.MethodType(forward_with_cache, layer)
