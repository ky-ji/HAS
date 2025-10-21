"""
LeRobot Format Data Collection Script with BAC Acceleration

This script collects episode data (videos + actions) from policy rollouts
and saves them in LeRobot-compatible format. Supports BAC (Block-wise Adaptive Caching)
for accelerated data collection.

Usage:
# Without BAC acceleration (original speed)
python scripts/collect_lerobot_data.py \
    --checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt\
    --output_dir datas/lerobot/kitchen \
    --num_episodes 100 \
    --device cuda:0

# With BAC acceleration (optimal caching)
python scripts/collect_lerobot_data.py \
    --checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt\
    --output_dir datas/lerobot/kitchen \
    --num_episodes 50 \
    --device cuda:2 \
    --use_bac \
    --cache_mode optimal \
    --optimal_steps_dir assets/kitchen/original/optimal_steps/cosine \
    --metric cosine \
    --num_caches 10 \
    --num_bu_blocks 3

# Acceleration (threshold caching)
python scripts/collect_lerobot_data.py \
    --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt\
    --output_dir datas/lerobot/can_ph\
    --num_episodes 50 \
    --device cuda:3 \
    --use_bac \
    --cache_mode threshold \
    --cache_threshold 5

python scripts/collect_lerobot_data.py \
    --checkpoint checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt\
    --output_dir datas/lerobot/tool_hang_ph \
    --num_episodes 50 \
    --device cuda:2 \
    --use_bac \
    --cache_mode optimal \
    --optimal_steps_dir assets/tool_hang_ph/original/optimal_steps/cosine \
    --metric cosine \
    --num_caches 5 \
    --num_bu_blocks 3


"""

import os
import sys
import pathlib
import click
import hydra
import torch
import dill
import numpy as np
import json
import logging
from typing import Dict, List
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.common.pytorch_util import dict_apply
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from copy import deepcopy

# Optional: pandas for parquet export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available, will save to JSON instead of Parquet")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lerobot_collector")


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class LeRobotDataCollector:
    """
    Collects episode data in LeRobot format.

    For each episode, collects:
    - Video frames from all available cameras
    - Action sequences at each timestep
    - Observation states (proprioceptive data)
    - Episode metadata (success, length, timestamps)
    """

    def __init__(self, output_dir: str):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.episodes_dir = self.output_dir / "episodes"
        self.episodes_dir.mkdir(exist_ok=True)

        # Storage for collected data
        self.episodes_data = []
        self.current_episode_idx = 0

    def start_episode(self, episode_idx: int):
        """Initialize storage for a new episode."""
        self.current_episode_idx = episode_idx
        self.current_episode_data = {
            'episode_idx': episode_idx,
            'frames': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'timestamps': []
        }
        self.current_frame_idx = 0

    def collect_step(self,
                     observation: Dict[str, np.ndarray],
                     action: np.ndarray,
                     reward: float,
                     timestamp: float = None):
        """
        Collect data for a single timestep.

        Args:
            observation: Dict with keys like 'agentview_image', 'robot0_eef_pos', etc.
            action: Action array taken at this step
            reward: Reward received
            timestamp: Optional timestamp (will auto-increment if None)
        """
        if timestamp is None:
            timestamp = self.current_frame_idx * 0.1  # Assume 10 Hz

        # Store frame data
        frame_data = {
            'episode_idx': self.current_episode_idx,
            'frame_idx': self.current_frame_idx,
            'timestamp': timestamp,
            'action': action.copy(),
            'reward': reward,
            'observation': {}
        }

        # Separate image observations from state observations
        for key, value in observation.items():
            if 'image' in key.lower() or 'camera' in key.lower():
                # Store image data
                frame_data['observation'][key] = value.copy()
            else:
                # Store state data (low-dim)
                frame_data['observation'][key] = value.copy()

        self.current_episode_data['frames'].append(frame_data)
        self.current_episode_data['actions'].append(action.copy())
        self.current_episode_data['observations'].append(observation)
        self.current_episode_data['rewards'].append(reward)
        self.current_episode_data['timestamps'].append(timestamp)

        self.current_frame_idx += 1

    def end_episode(self, success: bool = False, video_src_path: str = None):
        """Finalize and save the current episode."""
        episode_idx = self.current_episode_idx
        episode_dir = self.episodes_dir / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(exist_ok=True)

        # Save videos for each camera
        videos_dir = episode_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        # Organize frames by camera
        camera_frames = {}
        for frame_data in self.current_episode_data['frames']:
            for key, value in frame_data['observation'].items():
                if 'image' in key.lower() or 'camera' in key.lower():
                    if key not in camera_frames:
                        camera_frames[key] = []
                    camera_frames[key].append(value)

        # Save each camera's video
        video_paths = {}
        for camera_name, frames in camera_frames.items():
            video_path = self._save_video(frames, videos_dir, camera_name)
            video_paths[camera_name] = str(video_path.relative_to(self.output_dir))

        # If video was recorded by environment runner, copy it to episode directory
        if video_src_path and os.path.exists(video_src_path):
            import shutil
            video_dst_path = videos_dir / "episode_video.mp4"
            shutil.copy2(video_src_path, video_dst_path)
            video_paths['episode_video'] = str(video_dst_path.relative_to(self.output_dir))
            logger.info(f"Copied video from {video_src_path} to {video_dst_path}")

        # Prepare episode metadata
        episode_metadata = {
            'episode_idx': episode_idx,
            'length': len(self.current_episode_data['frames']),
            'success': bool(success),  # Convert to native Python bool
            'total_reward': float(sum(self.current_episode_data['rewards'])),  # Convert to native Python float
            'video_paths': video_paths,
            'timestamps': [float(t) for t in self.current_episode_data['timestamps']]  # Convert all timestamps
        }

        # Convert to JSON serializable format
        episode_metadata = convert_to_json_serializable(episode_metadata)

        # Save episode metadata
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(episode_metadata, f, indent=2)

        # Prepare flattened data for parquet/tabular format
        for frame_idx, frame_data in enumerate(self.current_episode_data['frames']):
            row = {
                'episode_idx': episode_idx,
                'frame_idx': frame_idx,
                'timestamp': frame_data['timestamp'],
                'success': success,
            }

            # Add action components
            action = frame_data['action']
            # Flatten action if it's multi-dimensional
            action_flat = action.flatten() if isinstance(action, np.ndarray) else action
            for i, action_val in enumerate(action_flat):
                row[f'action_{i}'] = float(action_val)

            # Add observation state (low-dim only, not images)
            for key, value in frame_data['observation'].items():
                if not ('image' in key.lower() or 'camera' in key.lower()):
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            row[f'obs_{key}'] = float(value)
                        else:
                            # Flatten multi-dimensional arrays
                            value_flat = value.flatten()
                            for i, obs_val in enumerate(value_flat):
                                row[f'obs_{key}_{i}'] = float(obs_val)
                    else:
                        row[f'obs_{key}'] = value

            # Add video paths
            for camera_name, video_path in video_paths.items():
                row[f'video_{camera_name}'] = video_path

            self.episodes_data.append(row)

        logger.info(f"Saved episode {episode_idx}: {len(self.current_episode_data['frames'])} frames, success={success}")

    def _save_video(self, frames: List[np.ndarray], output_dir: pathlib.Path, camera_name: str) -> pathlib.Path:
        """
        Save frames as MP4 video.

        Args:
            frames: List of numpy arrays (H, W, 3) with RGB images
            output_dir: Directory to save video
            camera_name: Name of the camera for filename

        Returns:
            Path to saved video file
        """
        video_path = output_dir / f"{camera_name}.mp4"

        if len(frames) == 0:
            logger.warning(f"No frames to save for {camera_name}")
            return video_path

        # Helper to convert any frame to 8-bit 3-channel BGR
        def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
            arr = img
            # Remove singleton channel dimension if present
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]

            # Convert dtype and scale to uint8
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    amin = float(np.nanmin(arr))
                    amax = float(np.nanmax(arr))
                    if amin >= -1.0 and amax <= 1.0:
                        # Handle [-1,1] or [0,1]
                        if amin < 0.0:
                            arr = (arr + 1.0) / 2.0
                        arr = np.clip(arr, 0.0, 1.0) * 255.0
                    else:
                        # Assume already in [0,255] range (or arbitrary) -> clip
                        arr = np.clip(arr, 0.0, 255.0)
                    arr = arr.astype(np.uint8)
                elif np.issubdtype(arr.dtype, np.integer):
                    if arr.dtype == np.uint16:
                        # Scale 16-bit to 8-bit
                        arr = (arr / 257.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            # Ensure 3-channel
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                if arr.shape[-1] == 4:
                    # Drop alpha
                    arr = arr[..., :3]
                elif arr.shape[-1] != 3:
                    # Fallback: reduce/expand to 3 channels
                    arr = arr[..., :3] if arr.shape[-1] > 3 else np.tile(arr, (1, 1, 3 // arr.shape[-1]))

            # Assume current is RGB, convert to BGR for OpenCV
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return bgr

        # Get video properties from first processed frame
        first_frame_bgr = to_bgr_uint8(frames[0])
        height, width = first_frame_bgr.shape[:2]
        fps = 10  # Default FPS, adjust as needed

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # Write frames
        for frame in frames:
            frame_bgr = to_bgr_uint8(frame)
            out.write(frame_bgr)

        out.release()
        logger.info(f"Saved video: {video_path} ({len(frames)} frames)")

        return video_path

    def save_dataset(self):
        """Save collected data as parquet file (LeRobot format)."""
        if not self.episodes_data:
            logger.warning("No episode data to save")
            return

        output_path = self.output_dir / "episodes_data.parquet"

        if PANDAS_AVAILABLE:
            # Save as Parquet (preferred LeRobot format)
            df = pd.DataFrame(self.episodes_data)
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved dataset to {output_path} (Parquet format)")
        else:
            # Fallback: save as JSON
            output_path = self.output_dir / "episodes_data.json"
            with open(output_path, 'w') as f:
                json.dump(self.episodes_data, f, indent=2)
            logger.info(f"Saved dataset to {output_path} (JSON format)")

        # Save summary statistics
        summary = {
            'total_episodes': len(set(row['episode_idx'] for row in self.episodes_data)),
            'total_frames': len(self.episodes_data),
            'successful_episodes': sum(1 for row in self.episodes_data if row.get('success', False) and row['frame_idx'] == 0),
        }

        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary: {summary}")


def collect_episodes_with_custom_runner(
    policy,
    env_runner,
    collector: LeRobotDataCollector,
    num_episodes: int,
    device: str = 'cuda:0'
):
    """
    Collect episodes by extending the existing runner with per-step data capture.

    This function manually runs the rollout loop and captures per-step data.
    """
    logger.info(f"Starting LeRobot data collection: {num_episodes} episodes")

    env = env_runner.env

    # Setup for single environment collection
    n_envs = len(env_runner.env_fns)
    n_inits = num_episodes
    n_chunks = math.ceil(n_inits / n_envs)

    episode_idx = 0

    # Get environment name for progress bar
    env_name = getattr(env_runner, 'env_meta', {}).get('env_name', 'Unknown')
    if env_name == 'Unknown':
        # Try to infer from class name
        runner_class_name = type(env_runner).__name__
        if 'Kitchen' in runner_class_name:
            env_name = 'Kitchen'
        elif 'BlockPush' in runner_class_name:
            env_name = 'BlockPush'
        elif 'Robomimic' in runner_class_name:
            env_name = 'Robomimic'
        else:
            env_name = runner_class_name

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end = min(n_inits, start + n_envs)
        this_n_active_envs = end - start

        # Get init functions for this chunk
        this_init_fns = env_runner.env_init_fn_dills[start:end]
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([env_runner.env_init_fn_dills[0]] * n_diff)

        # Initialize environments
        env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

        # Reset environment
        obs = env.reset()
        past_action = None
        policy.reset()

        # Track per-environment data
        env_episode_data = [[] for _ in range(this_n_active_envs)]

        pbar = tqdm(
            total=env_runner.max_steps,
            desc=f"Collect {env_name} {chunk_idx+1}/{n_chunks}",
            leave=False,
            mininterval=1.0
        )

        done = False
        step_count = 0

        while not done:
            # Create observation dict
            # Handle both dict and array observations
            try:
                # Try to use obs as a dict (for image-based environments)
                if isinstance(obs, dict):
                    np_obs_dict = dict(obs)
                else:
                    # For low-dim runners, obs is just an array
                    np_obs_dict = {'obs': obs.astype(np.float32)}
            except (ValueError, TypeError):
                # If dict conversion fails, treat as array
                np_obs_dict = {'obs': obs.astype(np.float32)}

            if env_runner.past_action and (past_action is not None):
                np_obs_dict['past_action'] = past_action[:,-(env_runner.n_obs_steps-1):].astype(np.float32)

            # Device transfer
            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device)
            )

            # Run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # Device transfer
            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().to('cpu').numpy()
            )

            action = np_action_dict['action']

            # Transform action if needed (for image runners with abs_action)
            env_action = action
            if hasattr(env_runner, 'abs_action') and env_runner.abs_action:
                env_action = env_runner.undo_transform_action(action)

            # Store step data for each active environment
            timestamp = step_count * 0.1  # Assume 10 Hz
            for env_idx in range(this_n_active_envs):
                step_data = {
                    'obs': {k: v[env_idx] for k, v in np_obs_dict.items()},
                    'action': action[env_idx],
                    'timestamp': timestamp
                }
                env_episode_data[env_idx].append(step_data)

            # Step environment
            obs, reward, done_flags, info = env.step(env_action)
            done = np.all(done_flags)
            past_action = action

            # Store rewards
            for env_idx in range(this_n_active_envs):
                if len(env_episode_data[env_idx]) > 0:
                    env_episode_data[env_idx][-1]['reward'] = reward[env_idx]

            step_count += 1
            pbar.update(action.shape[1])

        pbar.close()

        # Get final rewards and video paths
        all_rewards = env.call('get_attr', 'reward')[:this_n_active_envs]

        # Get video paths from the environment
        all_video_paths = []
        try:
            rendered_paths = env.render()[:this_n_active_envs]
            all_video_paths = rendered_paths
            logger.info(f"Retrieved {len(all_video_paths)} video paths from environment")
        except Exception as e:
            logger.warning(f"Could not retrieve video paths from environment: {e}")
            all_video_paths = [None] * this_n_active_envs

        # Process collected data for each environment
        for env_idx in range(this_n_active_envs):
            collector.start_episode(episode_idx)

            # Collect all steps
            for step_data in env_episode_data[env_idx]:
                collector.collect_step(
                    observation=step_data['obs'],
                    action=step_data['action'],
                    reward=step_data.get('reward', 0.0),
                    timestamp=step_data['timestamp']
                )

            # Determine success (threshold-based)
            success = np.max(all_rewards[env_idx]) > 0.9

            # Get video path for this episode and copy it to episode directory
            video_src_path = all_video_paths[env_idx]
            collector.end_episode(success=success, video_src_path=video_src_path)

            logger.info(
                f"Episode {episode_idx}: "
                f"{len(env_episode_data[env_idx])} steps, "
                f"reward={np.max(all_rewards[env_idx]):.3f}, "
                f"success={success}"
            )

            episode_idx += 1

        # Clear video buffer
        _ = env.reset()

    # Save collected dataset
    collector.save_dataset()
    logger.info(f"Data collection complete: {episode_idx} episodes saved")


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Model checkpoint path')
@click.option('-o', '--output_dir', required=True, help='Output directory for collected data')
@click.option('-d', '--device', default='cuda:0', help='Device to run on')
@click.option('--num_episodes', default=50, type=int, help='Number of episodes to collect')
@click.option('--dataset_path', default=None, help='Path to dataset (for env metadata)')
@click.option('--use_bac', is_flag=True, help='Enable BAC acceleration')
@click.option('--cache_mode', default='threshold', type=click.Choice(['original', 'threshold', 'optimal']), help='Caching mode (only used if --use_bac is set)')
@click.option('--cache_threshold', default=5, type=int, help='Cache threshold for threshold mode')
@click.option('--optimal_steps_dir', default=None, help='Directory for optimal schedules (required for optimal mode)')
@click.option('--num_caches', default=5, type=int, help='Number of cache updates in optimal schedule')
@click.option('--metric', default='cosine', help='Similarity metric for optimal schedules')
@click.option('--num_bu_blocks', default=0, type=int, help='Number of blocks for BU algorithm (0 disables BU)')
def main(checkpoint, output_dir, device, num_episodes, dataset_path,
         use_bac, cache_mode, cache_threshold, optimal_steps_dir,
         num_caches, metric, num_bu_blocks):
    """Collect robot data in LeRobot format with optional BAC acceleration."""
    logger.info("="*80)
    logger.info("LeRobot Data Collection Script")
    if use_bac:
        logger.info(f"BAC Acceleration: ENABLED (mode={cache_mode})")
    else:
        logger.info("BAC Acceleration: DISABLED")
    logger.info("="*80)

    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Get policy
    policy = workspace.model
    policy.to(device)
    policy.eval()
    logger.info(f"Policy loaded: {type(policy).__name__}")

    # Apply BAC acceleration if requested
    if use_bac:
        logger.info("\n" + "="*80)
        logger.info("Applying BAC Acceleration")
        logger.info("="*80)

        # Create a copy for acceleration
        policy = deepcopy(policy)

        if cache_mode == 'optimal':
            if optimal_steps_dir is None:
                raise ValueError("--optimal_steps_dir is required when using cache_mode=optimal")
            logger.info(f"Mode: Optimal caching")
            logger.info(f"Optimal steps directory: {optimal_steps_dir}")
            logger.info(f"Metric: {metric}")
            logger.info(f"Number of caches: {num_caches}")
            logger.info(f"BU blocks: {num_bu_blocks}")

            policy = FastDiffusionPolicy.apply_cache(
                policy=policy,
                cache_mode='optimal',
                optimal_steps_dir=optimal_steps_dir,
                num_caches=num_caches,
                metric=metric,
                num_bu_blocks=num_bu_blocks
            )
        elif cache_mode == 'threshold':
            logger.info(f"Mode: Threshold caching")
            logger.info(f"Cache threshold: {cache_threshold}")
            logger.info(f"BU blocks: {num_bu_blocks}")

            policy = FastDiffusionPolicy.apply_cache(
                policy=policy,
                cache_mode='threshold',
                cache_threshold=cache_threshold,
                num_bu_blocks=num_bu_blocks
            )
        else:  # original
            logger.info(f"Mode: Original (no caching)")
            policy = FastDiffusionPolicy.apply_cache(
                policy=policy,
                cache_mode='original',
                num_bu_blocks=num_bu_blocks
            )

        logger.info("BAC acceleration applied successfully")
        logger.info("="*80 + "\n")

    # Initialize data collector
    collector = LeRobotDataCollector(output_dir)

    # Create environment runner
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    env_runner_cfg['n_train'] = 0
    env_runner_cfg['n_test'] = num_episodes
    env_runner_cfg['n_train_vis'] = 0
    env_runner_cfg['n_test_vis'] = num_episodes  # Record all episodes

    env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)

    # Run data collection with LeRobot format
    logger.info("\n" + "="*80)
    logger.info("Starting LeRobot format data collection")
    logger.info("="*80 + "\n")

    try:
        collect_episodes_with_custom_runner(
            policy=policy,
            env_runner=env_runner,
            collector=collector,
            num_episodes=num_episodes,
            device=device
        )

        logger.info("\n" + "="*80)
        logger.info("Data collection completed successfully!")
        if use_bac:
            logger.info(f"BAC Acceleration: {cache_mode} mode")
            if cache_mode == 'optimal':
                logger.info(f"  - Optimal schedules: {optimal_steps_dir}")
                logger.info(f"  - Num caches: {num_caches}")
            elif cache_mode == 'threshold':
                logger.info(f"  - Cache threshold: {cache_threshold}")
            if num_bu_blocks > 0:
                logger.info(f"  - BU algorithm: {num_bu_blocks} blocks")
            logger.info("Expected speedup: 2-3x (with <5% accuracy loss)")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Episodes directory: {output_dir}/episodes/")
        logger.info(f"Dataset file: {output_dir}/episodes_data.parquet (or .json)")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
