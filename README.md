# BAC: Block-wise Adaptive Caching for Diffusion Policy

Official implementation of **BAC (Block-wise Adaptive Caching)**, a novel acceleration technique for Transformer-based Diffusion Policies in robotic manipulation tasks.

## Overview

Diffusion-based policies have shown remarkable performance in robotic manipulation, but their iterative denoising process leads to high computational costs during inference. BAC addresses this challenge by intelligently caching intermediate transformer activations across denoising steps, achieving significant speedups with minimal performance degradation.

### Key Features

- **Adaptive Caching Scheduler (ACS)**: Dynamically determines optimal cache update steps for each transformer block based on temporal activation similarity
- **Bubbling Union (BU) Algorithm**: Propagates cache schedules from deeper to shallower layers to maintain activation freshness
- **Block-wise Optimization**: Independently optimizes caching strategies for self-attention, cross-attention, and feedforward blocks
- **Minimal Accuracy Loss**: Achieves 2-3x speedup while maintaining >95% of original task success rates

## Installation

### Prerequisites

- Python 3.9
- CUDA 11.6 or higher
- Conda package manager

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/kyji/BAC.git
cd BAC

# Create and activate conda environment
conda env create -f conda_environment.yaml
conda activate robodiff

# Install the package
pip install -e .
```

## Quick Start

### 1. Download Data and Checkpoints

#### Robomimic Tasks (Can, Lift, Square, Transport, Tool Hang)

**Data Structure:**
```
data/robomimic/datasets/
├── can/
│   ├── ph/  # Proficient-Human demonstrations
│   └── mh/  # Multi-Human demonstrations
├── lift/
│   ├── ph/
│   └── mh/
├── square/
│   ├── ph/
│   └── mh/
├── transport/
│   ├── ph/
│   └── mh/
└── tool_hang/
    ├── ph/
    └── mh/
```

**Checkpoint Structure:**
```
checkpoint/
├── can_ph/diffusion_policy_transformer/
│   ├── train_0/checkpoints/
│   │   ├── epoch=0650-test_mean_score=1.000.ckpt
│   │   └── latest.ckpt
│   ├── train_1/checkpoints/
│   │   └── ...
│   └── train_2/checkpoints/
│       └── ...
├── can_mh/diffusion_policy_transformer/
│   └── ...
└── ... (similar structure for other tasks)
```

#### Low-dimensional Tasks (Block Pushing, Kitchen)

**Checkpoint Structure:**
```
checkpoint/low_dim/
├── block_pushing/diffusion_policy_transformer/
│   ├── train_0/checkpoints/
│   │   ├── epoch=7550-test_mean_score=1.000.ckpt
│   │   └── latest.ckpt
│   ├── train_1/checkpoints/
│   └── train_2/checkpoints/
└── kitchen/diffusion_policy_transformer/
    ├── train_0/checkpoints/
    ├── train_1/checkpoints/
    └── train_2/checkpoints/
```

**Note:** We provide 3 training seeds (train_0, train_1, train_2) for each task. Results are reported as:
- **(max performance)** from `epoch=XXXX-test_mean_score=Y.YYY.ckpt`
- **(average of last 10 checkpoints)** from `latest.ckpt`

Download datasets and checkpoints from the original [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/data/) project or train your own models.

### 2. Compute Optimal Cache Schedules

Before running accelerated inference, compute the optimal caching schedules:

```bash
# For a single task
python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py \
  -c checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0650-test_mean_score=1.000.ckpt \
  -o assets/can_ph \
  -d cuda:0 \
  --num_caches 5,8,10,20 \
  --force_recompute

# Batch process all tasks
./shs/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5

# Task-specific batches
./shs/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5 --ph_only    # PH tasks only
./shs/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5 --mh_only    # MH tasks only
./shs/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5 --lowdim_only # Low-dim tasks
./shs/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5 --task can_ph # Specific task
```

### 3. Run Evaluation

Evaluate the policy with BAC acceleration:

```bash
# With BAC (optimal caching + BU algorithm)
python scripts/eval_fast_diffusion_policy.py \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0650-test_mean_score=1.000.ckpt \
  -o results/can_ph/bac \
  --device cuda:0 \
  --cache_mode optimal \
  --optimal_steps_dir assets/can_ph/optimal_steps/cosine \
  --metric cosine \
  --num_caches 5 \
  --num_bu_blocks 3 \
  --skip_video

# Baseline (no caching)
python scripts/eval_fast_diffusion_policy.py \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0650-test_mean_score=1.000.ckpt \
  -o results/can_ph/baseline \
  --device cuda:0 \
  --cache_mode original \
  --skip_video

# Threshold-based caching (uniform schedule)
python scripts/eval_fast_diffusion_policy.py \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0650-test_mean_score=1.000.ckpt \
  -o results/can_ph/threshold \
  --device cuda:0 \
  --cache_mode threshold \
  --cache_threshold 5 \
  --skip_video
```

### 4. Run Full Benchmarks

Run comprehensive benchmarks across all tasks:

```bash
# All benchmarks
./shs/benchmark/run_all_benchmarks.sh --device cuda:0 --skip_video

# Specific task categories (if scripts exist)
# ./shs/benchmark/run_bac_model_ph.sh --device cuda:0 --skip_video  # Proficient-Human tasks
# ./shs/benchmark/run_bac_model_mh.sh --device cuda:0 --skip_video  # Multi-Human tasks
# ./shs/benchmark/run_bac_model_bp.sh --device cuda:0 --skip_video  # Block Pushing
# ./shs/benchmark/run_bac_model_kitchen.sh --device cuda:0 --skip_video  # Kitchen
```

### 5. Collect Data in LeRobot Format

BAC includes a data collection script that exports policy rollouts in [LeRobot](https://github.com/huggingface/lerobot) format, making it easy to share and reuse collected demonstrations. **The script supports BAC acceleration for faster data collection.**

```bash
# Without BAC acceleration (original speed)
python scripts/collect_lerobot_data.py \
  --checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
  --output_dir datas/lerobot/kitchen \
  --num_episodes 100 \
  --device cuda:0

# With BAC acceleration (optimal caching) - 2-3x faster!
python scripts/collect_lerobot_data.py \
  --checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
  --output_dir datas/lerobot/kitchen \
  --num_episodes 100 \
  --device cuda:0 \
  --use_bac \
  --cache_mode optimal \
  --optimal_steps_dir assets/kitchen/optimal_steps/cosine \
  --metric cosine \
  --num_caches 5 \
  --num_bu_blocks 3

# With BAC acceleration (threshold caching)
python scripts/collect_lerobot_data.py \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
  --output_dir datas/lerobot/can_ph \
  --num_episodes 50 \
  --device cuda:3 \
  --use_bac \
  --cache_mode threshold \
  --cache_threshold 5

# Other vision-based tasks
python scripts/collect_lerobot_data.py \
  --checkpoint checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
  --output_dir datas/lerobot/square_ph \
  --num_episodes 100 \
  --device cuda:3 \
  --use_bac \
  --cache_mode optimal \
  --optimal_steps_dir assets/square_ph/optimal_steps/cosine \
  --num_caches 5 \
  --num_bu_blocks 3
```

#### LeRobot Data Format

The collected data follows the LeRobot specification for compatibility with HuggingFace datasets:

```
datas/lerobot/{task}/
├── episodes/                           # Per-episode data
│   ├── episode_000000/
│   │   ├── videos/
│   │   │   └── episode_video.mp4      # Episode video recording
│   │   └── metadata.json               # Episode metadata
│   ├── episode_000001/
│   │   └── ...
│   └── ...
├── episodes_data.parquet               # Consolidated tabular data
└── dataset_summary.json                # Dataset statistics
```

**episodes_data.parquet structure:**
- **Metadata columns**: `episode_idx`, `frame_idx`, `timestamp`, `success`
- **Action columns**: `action_0` through `action_N` (flattened action sequences)
- **Observation columns**: `obs_obs_0` through `obs_obs_M` (low-dimensional states)
- **Video columns**: `video_episode_video` (path to episode video)

**Example metadata.json:**
```json
{
  "episode_idx": 0,
  "length": 35,
  "success": true,
  "total_reward": 30.0,
  "video_paths": {
    "episode_video": "episodes/episode_000000/videos/episode_video.mp4"
  },
  "timestamps": [0.0, 0.1, 0.2, ...]
}
```

**Example dataset_summary.json:**
```json
{
  "total_episodes": 100,
  "total_frames": 3500,
  "successful_episodes": 95
}
```

#### Data Collection Features

- **BAC Acceleration Support**: Optional 2-3x speedup for faster data collection with minimal accuracy impact
- **Per-episode organization**: Each episode has isolated video and metadata
- **Aligned action-video data**: Actions are synchronized with video frames at 10Hz
- **Flattened observations and actions**: Multi-dimensional arrays are flattened for Parquet compatibility
- **Success tracking**: Episode success is computed based on reward thresholds
- **Parquet export**: Efficient columnar storage for large-scale datasets
- **JSON fallback**: Automatically uses JSON if pandas is unavailable

**BAC Acceleration Options:**
- `--use_bac`: Enable BAC acceleration
- `--cache_mode`: Choose `optimal` (adaptive), `threshold` (uniform), or `original` (no caching)
- `--cache_threshold`: Cache update interval for threshold mode (default: 5)
- `--optimal_steps_dir`: Path to precomputed optimal schedules (required for optimal mode)
- `--num_caches`: Number of cache updates for optimal schedules (default: 5)
- `--num_bu_blocks`: Number of blocks for Bubbling Union algorithm (default: 0)
- `--metric`: Similarity metric for optimal schedules (default: cosine)


## Project Structure

```
BAC/
├── diffusion_policy/
│   ├── acceleration/              # BAC caching wrapper implementation
│   ├── activation_utils/          # Activation analysis and optimal schedule computation
│   ├── activation_visualization/  # Plotting and visualization tools
│   ├── model/                     # Transformer and diffusion model architectures
│   ├── env/                       # Simulation environments (robosuite, etc.)
│   └── dataset/                   # Dataset loaders
├── scripts/
│   ├── eval_fast_diffusion_policy.py  # Main evaluation script
│   └── collect_lerobot_data.py        # LeRobot format data collection
├── shs/                           # Shell scripts for batch experiments
│   ├── benchmark/                 # Benchmark automation scripts
│   └── get_steps/                 # Scripts for computing optimal schedules
├── checkpoint/                    # Pre-trained model checkpoints
├── data/                          # Training datasets (robomimic, etc.)
├── datas/                         # Collected demonstration data (LeRobot format)
│   └── lerobot/                   # LeRobot formatted datasets
│       ├── kitchen/
│       ├── can_ph/
│       ├── lift_ph/
│       └── ...
├── assets/                        # Cached activations and optimal schedules
│   └── {task}/
│       ├── original/              # Activations and similarity matrices
│       └── optimal_steps/         # Per-module optimal cache schedules
└── results/                       # Evaluation results
    └── {task}/{mode}/
        ├── benchmark_results.json
        └── eval_results.json
```

## Methodology

### Adaptive Caching Scheduler (ACS)

BAC analyzes the temporal similarity of activations across denoising steps to identify when cached values can be safely reused:

1. **Activation Collection**: Hook into transformer blocks during inference to collect intermediate activations
2. **Similarity Analysis**: Compute pairwise similarity matrices (cosine, L1, MSE) between activations at different timesteps
3. **Dynamic Programming**: Optimize cache update schedules to minimize similarity error while constraining the number of cache updates
4. **Per-Block Schedules**: Generate independent schedules for self-attention, cross-attention, and feedforward blocks

### Bubbling Union (BU) Algorithm

The BU algorithm ensures activation freshness by propagating cache update steps:

1. Identify blocks with highest caching error
2. Propagate cache update steps from deeper (later) layers to shallower (earlier) layers


## Supported Tasks

### Vision-based Manipulation (robomimic)

- **Proficient-Human (PH)**: Can, Lift, Square, Transport, Tool Hang
- **Miexed-Human (MH)**: Mixed Proficient and Non-Proficient Data

### Low-dimensional State-based

- **Block Pushing**: Continuous control task
- **Kitchen**: Multi-stage manipulation task

## Evaluation Metrics

Results are reported in `results/{task}/{mode}/eval_results.json`:

- `mean_score`: Task success rate (0-1)
- `speedup`: Inference time speedup compared to baseline
- `flops`: Computational cost in FLOPs
- `frequency`: Actions per second



## Citation

If you find this work useful, please cite:

```bibtex
@article{ji2025block,
  title={Block-wise Adaptive Caching for Accelerating Diffusion Policy},
  author={Ji, Kangye and Meng, Yuan and Cui, Hanyun and Li, Ye and Hua, Shengjia and Chen, Lei and Wang, Zhi},
  journal={arXiv preprint arXiv:2506.13456},
  year={2025}
}
```

## Acknowledgments

This work builds upon:
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Original diffusion policy implementation
- [robomimic](https://github.com/ARISE-Initiative/robomimic) - Robot learning framework and benchmarks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [jky25@mails.tsinghua.edu.cn]
