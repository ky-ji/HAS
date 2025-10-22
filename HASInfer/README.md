# HASInfer - Hash-based Acceleration System Inference Module

HASInfer 是一个用于加速扩散策略推理的插件式模块，采用**基于哈希的线性多步法**（Hash-based Linear Multistep Methods）技术，结合块级自适应缓存实现高效推理加速。

## 核心技术

HASInfer 将**数值分析中的线性多步法**应用于扩散模型加速：

- **Adams-Bashforth (AB)**：显式预测方法，使用历史导数外推未来状态

通过离线预计算导数和在线哈希表查询，实现快速且准确的特征预测。

## 目录结构

```
HASInfer/
├── __init__.py                              # 模块初始化
├── README.md                                # 本文档
├── core/                                    # 核心加速功能
│   ├── __init__.py
│   ├── diffusion_hash_wrapper_multistep.py # FastDiffusionPolicyMultistep 实现
│   └── hash_util_multistep.py              # 哈希表实现
├── analysis/                                # 激活分析工具
│   ├── __init__.py
│   ├── collect_activations.py              # 激活收集
│   ├── activation_analysis.py              # 激活分析
│   ├── compute_derivatives.py              # 导数计算（五点差分法）
│   ├── get_optimal_cache_update_steps.py   # 获取最优缓存步骤
│   ├── optimal_cache_scheduler.py          # 最优缓存调度器
│   ├── bu_block_selection.py               # BU块选择算法
│   └── run_policy.py                       # 策略运行接口
└── scripts/                                 # 评估脚本
    ├── __init__.py
    └── eval_hash_diffusion_policy.py        # HAS加速策略评估
```

## 快速开始

### 1. 基本使用

```python
from HASInfer.core.diffusion_hash_wrapper_multistep import FastDiffusionPolicyMultistep

# 加载你的策略
policy = workspace.model

# 应用 HAS 加速（最优模式 + Adams-Bashforth）
fast_policy = FastDiffusionPolicyMultistep.apply_cache(
    policy=policy,
    cache_mode='optimal',
    optimal_steps_dir='assets/task_name/original/optimal_steps/cosine',
    num_caches=7,
    metric='cosine',
    num_bu_blocks=5,
    multistep_method='ab',              # Adams-Bashforth
    max_queue_length=3,                 # 历史队列长度
    precomputed_derivatives_path='assets/task_name/derivatives/derivatives_five_point_stencil.pkl'
)

# 使用加速后的策略进行推理
action = fast_policy.predict_action(obs_dict)
```

### 2. 线性多步法选项

HASInfer 支持AB线性多步法：

#### (1) Adams-Bashforth (AB) - 推荐
```python
FastDiffusionPolicyMultistep.apply_cache(
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
```


### 3. 获取最优缓存步骤

```bash
python HASInfer/analysis/get_optimal_cache_update_steps.py \
    -c checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt \
    -o assets/lift_ph \
    -d cuda:2 \
    --num_caches '5,7,10,20' \
    --force_recompute
```

### 4. 离线预计算导数

在使用线性多步法之前，建议先预计算导数以提高在线推理速度：

```bash
# 使用五点差分法计算导数
python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/kitchen/original/activations.pkl \
    --optimal_steps_dir assets/kitchen/original/optimal_steps/cosine \
    --output_dir assets/kitchen/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5
```


### 5. 评估加速策略

```bash
python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:2 \
--cache_mode optimal \
--optimal_steps_dir assets/kitchen/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/kitchen/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video
```

**注意**：如果遇到 `BrokenPipeError`，请使用 `--n_envs` 参数限制并行环境数量（推荐值：8-16）。

## 核心功能

### FastDiffusionPolicyMultistep

核心加速类，结合了：
- **线性多步法**：使用历史激活值和导数预测当前特征
- **哈希表缓存**：快速存储和查询历史激活
- **自适应调度**：基于相似度的智能缓存更新策略
- **BU算法**：选择高误差块进行重点优化

### GridBasedHashTable_AdamsBashforth

哈希表实现，支持：
- **历史队列管理**：自动维护固定长度的历史记录
- **多步法系数缓存**：预计算并缓存 AB/ABM/BDF 系数
- **拉格朗日插值**：线性和二次插值提高预测精度
- **设备优化**：自动管理 GPU/CPU 数据传输

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `multistep_method` | 线性多步法类型 | `'ab'` (Adams-Bashforth) |
| `max_queue_length` | 历史队列长度 | `3`（2-4之间） |
| `num_caches` | 缓存更新次数 | `7`（总步数的 10-20%） |
| `num_bu_blocks` | BU算法块数 | `5`（0表示禁用） |
| `precomputed_derivatives_path` | 预计算导数路径 | 建议提供以提升速度 |

## 工作流程

### 离线阶段

1. **收集激活值**：运行原始模型，记录每个扩散步的激活
2. **计算导数**：使用五点差分法等数值方法计算一阶导数
3. **分析相似度**：计算步间激活相似度矩阵
4. **确定缓存点**：基于相似度选择最优缓存更新步骤
5. **选择BU块**：识别高误差块用于重点优化

### 在线阶段

1. **初始步骤**：正常前向传播，填充历史队列
2. **缓存命中**：
   - 查询哈希表获取历史激活
   - 加载对应的预计算导数
   - 应用线性多步法预测当前激活
   - 使用拉格朗日插值提高精度
3. **缓存更新**：按最优步骤更新缓存
4. **BU优化**：高误差块仍执行完整前向传播

## 高级用法


### 激活收集与分析

```bash
# 收集原始模型的激活（用于后续分析）
python -m HASInfer.analysis.collect_activations \
    -c checkpoint/model.ckpt \
    -o visualization \
    -t task_name \
    -d cuda:0 \
    --demo_idx 0 \
    --cache_mode original

# 收集加速模型的激活（用于对比）
python -m HASInfer.analysis.collect_activations \
    -c checkpoint/model.ckpt \
    -o visualization \
    -t task_name \
    -d cuda:0 \
    --demo_idx 0 \
    --cache_mode optimal \
    --optimal_steps_dir assets/task_name/original/optimal_steps/cosine \
    --num_caches 7 \
    --metric cosine \
    --multistep_method ab \
    --max_queue_length 3
```

### 块误差分析

```bash
python -m HASInfer.analysis.bu_block_selection \
    -o visualization \
    -t task_name \
    --cache_mode original \
    --num_blocks 5
```

## 技术细节

### 线性多步法原理

#### Adams-Bashforth (k阶)
$$x_{n+1} = x_n + h \sum_{j=0}^{k-1} \beta_j f(t_{n-j}, x_{n-j})$$

其中 $\beta_j$ 是 AB 系数：
- 1阶：`[1.0]`
- 2阶：`[3/2, -1/2]`
- 3阶：`[23/12, -16/12, 5/12]`
- 4阶：`[55/24, -59/24, 37/24, -9/24]`

#### 拉格朗日插值

对于不在缓存点的时间步，使用拉格朗日插值：

**线性插值（2点）**：
$$x(t) = \frac{t-j}{i-j} \cdot X_i + \frac{t-i}{j-i} \cdot X_j$$

**二次插值（3点）**：
$$x(t) = \frac{jk}{(i-j)(i-k)} X_i + \frac{ik}{(j-i)(j-k)} X_j + \frac{ij}{(k-i)(k-j)} X_k$$
