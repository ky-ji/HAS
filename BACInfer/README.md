# BACInfer - Block-wise Adaptive Caching Inference Module

BACInfer 是一个用于加速扩散策略推理的插件式模块，采用块级自适应缓存（Block-wise Adaptive Caching, BAC）技术。

## 目录结构

```
BACInfer/
├── __init__.py                      # 模块初始化
├── README.md                        # 本文档
├── core/                            # 核心加速功能
│   ├── __init__.py
│   └── diffusion_cache_wrapper.py  # FastDiffusionPolicy 实现
├── analysis/                        # 激活分析工具
│   ├── __init__.py
│   ├── collect_activations.py      # 激活收集
│   ├── activation_analysis.py      # 激活分析
│   ├── optimal_cache_scheduler.py  # 最优缓存调度器
│   ├── get_optimal_cache_update_steps.py  # 获取最优缓存步骤
│   ├── bu_block_selection.py       # BU块选择算法
│   └── run_policy.py               # 策略运行接口
├── scripts/                         # 评估和数据收集脚本
│   ├── __init__.py
│   ├── eval_fast_diffusion_policy.py       # BAC加速策略评估
│   └── collect_lerobot_data.py             # LeRobot格式数据收集
└── shell/                           # Shell脚本
    ├── main/                        # 主要评估脚本
    │   ├── run_bac_all.sh           # BAC批量评估
    │   ├── ablation_bac.sh          # 消融实验
    │   └── run_uniform_all.sh       # 均匀缓存评估
    └── get_steps/                   # 获取最优步骤脚本
        ├── get_optimal_steps.sh     # 获取最优缓存步骤
        ├── get_practical_steps.sh   # 获取实用步骤
        └── get_steps_info.sh         # 步骤信息查看
```

## 快速开始

### 1. 基本使用

```python
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy

# 加载你的策略
policy = workspace.get_policy()

# 应用BAC加速（最优模式）
policy = FastDiffusionPolicy.apply_cache(
    policy=policy,
    cache_mode='optimal',
    optimal_steps_dir='assets/task_name/original/optimal_steps/cosine',
    num_caches=10,
    metric='cosine',
    num_bu_blocks=3
)

# 使用加速后的策略进行推理
action = policy.predict_action(obs_dict)
```

### 2. 三种缓存模式

#### (1) Original 模式 - 无缓存（基线）
```python
FastDiffusionPolicy.apply_cache(policy, cache_mode='original')
```

#### (2) Threshold 模式 - 固定间隔缓存
```python
FastDiffusionPolicy.apply_cache(
    policy, 
    cache_mode='threshold',
    cache_threshold=5  # 每5步更新一次缓存
)
```

#### (3) Optimal 模式 - 自适应缓存（推荐）
```python
FastDiffusionPolicy.apply_cache(
    policy,
    cache_mode='optimal',
    optimal_steps_dir='assets/task_name/original/optimal_steps/cosine',
    num_caches=10,
    metric='cosine',
    num_bu_blocks=3
)
```

### 3. 获取最优缓存步骤

在使用 optimal 模式之前，需要先计算最优缓存步骤：

```bash
# Python方式
python -m BACInfer.analysis.get_optimal_cache_update_steps \
    -c checkpoint/task_name/model.ckpt \
    -o assets/task_name \
    -d cuda:0 \
    --num_caches 10 \
    --metrics cosine

# Shell脚本方式
bash BACInfer/shell/get_steps/get_optimal_steps.sh \
    --device cuda:0 \
    --num_caches 10 \
    --metrics cosine \
    --task task_name
```

### 4. 评估加速策略

```bash
python -m BACInfer.scripts.eval_fast_diffusion_policy \
    --checkpoint checkpoint/task_name/model.ckpt \
    --output_dir results/task_name/bac \
    --device cuda:0 \
    --cache_mode optimal \
    --optimal_steps_dir assets/task_name/original/optimal_steps/cosine \
    --metric cosine \
    --num_caches 10 \
    --num_bu_blocks 3 \
    --skip_video
```

Example:
```bash
python -m BACInfer.scripts.eval_fast_diffusion_policy \
    --checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
    --output_dir results/debug/bac \
    --device cuda:0 \
    --cache_mode optimal \
    --optimal_steps_dir assets/kitchen/original/optimal_steps/cosine \
    --metric cosine \
    --num_caches 10 \
    --num_bu_blocks 5 \
    --skip_video

python -m BACInfer.scripts.eval_fast_diffusion_policy \
    --checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
    --output_dir results/debug/bac \
    --device cuda:0 \
    --cache_mode threshold \
    --num_caches 10 \
    --skip_video

```




### 5. 批量评估

```bash
bash BACInfer/shell/main/run_bac_all.sh \
    --device cuda:0 \
    --steps 10 \
    --metric cosine \
    --seeds 0,1,2 \
    --ph_only
```

## 核心功能

### FastDiffusionPolicy

核心加速类，提供三种缓存模式：

- **original**: 无缓存，作为基线对比
- **threshold**: 固定间隔缓存更新
- **optimal**: 基于激活相似度的自适应缓存调度

### BU (Bubbling Union) 算法

通过分析块级误差，选择需要频繁更新的高误差块，进一步优化缓存策略。

参数 `num_bu_blocks` 控制应用BU算法的块数：
- `0`: 禁用BU，对每个块独立使用自适应步骤
- `>0`: 启用BU，选择指定数量的高误差块进行步骤传播



确保原项目的 `diffusion_policy` 在Python路径中可访问。

## 高级用法

### 激活收集与分析

```bash
# 收集原始模型的激活
python -m BACInfer.analysis.collect_activations \
    -c checkpoint/model.ckpt \
    -o visualization \
    -t task_name \
    -d cuda:0 \
    --demo_idx 0 \
    --cache_mode original

# 收集缓存模型的激活（用于比较）
python -m BACInfer.analysis.collect_activations \
    -c checkpoint/model.ckpt \
    -o visualization \
    -t task_name \
    -d cuda:0 \
    --demo_idx 0 \
    --cache_mode optimal \
    --optimal_steps_dir assets/task_name/original/optimal_steps/cosine \
    --num_caches 10 \
    --metric cosine \
    --bu
```


### 块误差分析

```bash
python -m BACInfer.analysis.bu_block_selection \
    -o visualization \
    -t task_name \
    --cache_mode original \
    --num_blocks 5
```


