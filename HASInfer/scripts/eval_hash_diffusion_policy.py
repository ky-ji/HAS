"""
使用示例（修改对应文件路径）
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

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:3 \
--cache_mode optimal \
--optimal_steps_dir assets/block_pushing/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/block_pushing/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:2 \
--cache_mode optimal \
--optimal_steps_dir assets/pusht/original/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/pusht/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:2 \
--cache_mode optimal \
--optimal_steps_dir assets/lift_ph/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/lift_ph/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:3 \
--cache_mode optimal \
--optimal_steps_dir assets/can_ph/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/can_ph/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:3 \
--cache_mode optimal \
--optimal_steps_dir assets/square_ph/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/square_ph/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:3 \
--cache_mode optimal \
--optimal_steps_dir assets/transport_ph/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/transport_ph/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

python HASInfer/scripts/eval_hash_diffusion_policy.py \
--checkpoint checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt \
-o results/multistep_ab \
--device cuda:1 \
--cache_mode optimal \
--optimal_steps_dir assets/tool_hang_ph/original/optimal_steps/cosine \
--metric cosine \
--num_caches 7 \
--num_bu_blocks 5 \
--multistep_method ab \
--max_queue_length 3 \
--precomputed_derivatives_path assets/tool_hang_ph/original/derivatives/derivatives_five_point_stencil.pkl \
--skip_video

"""

import sys
# 使用行缓冲模式
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import numpy as np
import time
import json
import logging
from omegaconf import OmegaConf
from copy import deepcopy

# 导入thop库计算FLOPs

from thop import profile

import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import caching wrapper
from HASInfer.core.diffusion_hash_wrapper_multistep import FastDiffusionPolicyMultistep
from diffusion_policy.common.pytorch_util import dict_apply

# 配置基础日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_script")

@click.command()
@click.option('-c', '--checkpoint', required=True, help='模型检查点路径')
@click.option('-o', '--output_dir', required=True, help='输出目录')
@click.option('-d', '--device', default='cuda:0', help='运行设备')
@click.option('--cache_threshold', default=5, type=int, help='缓存阈值，每隔多少步更新一次缓存')
@click.option('--optimal_steps_dir', default=None, help='最优步骤目录路径，如果提供则使用最优步骤模式')
@click.option('--num_caches', default=30, type=int, help='缓存更新次数，用于最优步骤文件')
@click.option('--metric', default='cosine', help='相似度指标类型，用于加载最优步骤文件')
@click.option('--cache_mode', default=None, 
              type=click.Choice(['original', 'threshold', 'optimal', 'fix', 'propagate', 'edit', 'random']), 
              help='缓存模式，可选 original, threshold, optimal, fix, propagate, edit, random')
@click.option('--edit_steps', default=None, help='Custom steps for edit mode, comma-separated values')
@click.option('--skip_video', is_flag=True, help='是否跳过视频渲染')
@click.option('--num_bu_blocks', default=5, type=int, help='要应用BU算法的块数量，为0时禁用BU算法')
@click.option('--interpolation_ratio', default=1.0, type=float, help='Interpolation ratio between cached and original activations')
@click.option('--reference_activations_path', default=None, help='Path to reference activations')
@click.option('--precomputed_weights_path', default=None, help='Path to precomputed weights for similarity-based caching')
@click.option('--precomputed_derivatives_path', default=None, help='Path to precomputed derivatives for multistep methods')
@click.option('--multistep_method', default='ab', type=click.Choice(['abm', 'ab', 'bdf']), help='Linear multistep method: abm (Adams-Bashforth-Moulton), ab (Adams-Bashforth), bdf (Backward Differentiation Formula)')
@click.option('--max_queue_length', default=3, type=int, help='Maximum history queue length for multistep methods (2-4 recommended)')

def main(checkpoint, output_dir, device, cache_threshold, optimal_steps_dir, 
         num_caches, metric, cache_mode, edit_steps, skip_video, num_bu_blocks, 
         interpolation_ratio, reference_activations_path, precomputed_weights_path,
         precomputed_derivatives_path, multistep_method, max_queue_length):
    # if os.path.exists(output_dir):
    #     click.confirm(f"输出路径 {output_dir} 已存在！是否覆盖?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载检查点
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    logger.info(f"配置已加载: {cfg._target_}")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    logger.info("工作区加载完成")
    
    # 获取原始策略
    original_policy = workspace.model
    original_policy.to(device)
    original_policy.eval()
    
    # 记录策略信息
    has_model = hasattr(original_policy, 'model')
    logger.info(f"策略类型: {type(original_policy).__name__}")
    if has_model:
        logger.info(f"model 类型: {type(original_policy.model).__name__}")
    
    # 获取动作帧数量
    actions_per_inference = original_policy.n_action_steps
    logger.info(f"每次推理的动作帧数量: {actions_per_inference}")
    
    # 创建快速策略实例
    logger.info("创建策略副本用于缓存加速...")
    fast_policy = deepcopy(original_policy)
    
    # 解析edit步骤（如果提供）
    edit_steps_list = None
    if edit_steps is not None:
        edit_steps_list = [int(s) for s in edit_steps.split(',')]
        logger.info(f"Edit steps: {edit_steps_list}")
    
    # 应用缓存加速（使用线性多步法）
    if cache_mode == 'optimal':
        logger.info(f"应用基于最优步骤的缓存加速（线性多步法）")
        logger.info(f"  - 步骤目录: {optimal_steps_dir}")
        logger.info(f"  - 指标: {metric}")
        logger.info(f"  - 缓存次数: {num_caches}")
        logger.info(f"  - 多步法: {multistep_method.upper()}")
        logger.info(f"  - 历史队列长度: {max_queue_length}")
        logger.info(f"  - BU算法块数: {num_bu_blocks}")
        if precomputed_derivatives_path:
            logger.info(f"  - 预计算导数: {precomputed_derivatives_path}")
        else:
            logger.info(f"  - 预计算导数: 未提供，将使用在线有限差分估算")
        
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
    else:  # threshold或original
        logger.info(f"应用基于阈值的缓存加速，阈值={cache_threshold}")
        fast_policy = FastDiffusionPolicyMultistep.apply_cache(
            policy=fast_policy,
            cache_mode=cache_mode,  # 可能是None、'threshold'或'original'
            cache_threshold=cache_threshold,
            num_bu_blocks=num_bu_blocks
        )
    
    # 生成模拟输入进行推理测试
    B = 1  # 批量大小
    To = cfg.n_obs_steps  # 观察步骤数
    
    # 检查是否为低维模型
    if hasattr(cfg, 'shape_meta'):
        logger.info("检测到图像模型配置")
        # 为每种观察类型创建张量
        obs_dict = {}
        for key, shape in cfg.shape_meta['obs'].items():
            # 调整形状以适应批量大小和观察步骤
            tensor_shape = [B, To] + list(shape['shape'])
            obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
    else:
        logger.info("检测到低维模型配置")
        # 使用obs_dim创建低维观察输入
        obs_dim = cfg.obs_dim if hasattr(cfg, 'obs_dim') else None
        
        # 如果没有明确定义obs_dim，尝试从任务配置获取
        if obs_dim is None and hasattr(cfg, 'task'):
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            if 'obs_dim' in task_cfg:
                obs_dim = task_cfg['obs_dim']
                logger.info(f"使用配置中的obs_dim: {obs_dim}")
        
        # 创建观察字典
        obs_dict = {
            'obs': torch.zeros((B, To, obs_dim), device=device, dtype=torch.float32)
        }
        
        # 处理可能的past_action
        if hasattr(cfg, 'use_past_action') and cfg.use_past_action:
            action_dim = cfg.action_dim
            obs_dict['past_action'] = torch.zeros((B, To, action_dim), device=device, dtype=torch.float32)

    # 记录创建的输入信息
    logger.info(f"创建的输入字典包含以下键: {list(obs_dict.keys())}")
    for key, tensor in obs_dict.items():
        logger.info(f"  {key}: shape={tensor.shape}")
    
    # 预热和测量原始策略的时间
    logger.info("预热...")
    try:
        with torch.no_grad(): 
            original_policy.predict_action(obs_dict)  #预测生成机器人行为
            #fast_policy.reset_cache()
            fast_policy.predict_action(obs_dict)
        logger.info("预热完成")
    except Exception as e:
        logger.error(f"预热时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 计算FLOPs（使用thop库）
    flops_value = 0.0

    logger.info("\n使用thop计算策略的FLOPs...")
    try:
        # 在模型预测前创建输入的副本，避免干扰
        obs_dict_copy = {}
        for key, value in obs_dict.items():
            obs_dict_copy[key] = value.clone()
            
        # 准备模型需要的输入参数
        # 1. 从输入中提取特征以创建样本和条件
        with torch.no_grad():
            # 标准化输入
            nobs = fast_policy.normalizer.normalize(obs_dict_copy)
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]
            
            # 确保获取正确的action_dim和obs_feature_dim
            if hasattr(fast_policy, 'action_dim'):
                Da = fast_policy.action_dim
            elif hasattr(cfg, 'action_dim'):
                Da = cfg.action_dim
            else:
                Da = cfg.task.action_dim
            
            if hasattr(fast_policy, 'obs_feature_dim'):
                Do = fast_policy.obs_feature_dim
            else:
                # 对于没有obs_feature_dim的低维模型，使用obs_dim
                if hasattr(cfg, 'obs_dim'):
                    Do = cfg.obs_dim
                else:
                    Do = cfg.task.obs_dim
            
            logger.info(f"Action dimension: {Da}, Observation feature dimension: {Do}")
            
            # 创建必要的输入
            device = fast_policy.device
            dtype = torch.float32
            timestep = torch.zeros(B, dtype=torch.long, device=device)  # 初始时间步为0
            
            # 处理条件输入
            cond = None
            sample = None
            
            # 获取动作维度和观察维度
            action_dim = fast_policy.action_dim
            obs_feature_dim = fast_policy.obs_dim if hasattr(fast_policy, 'obs_dim') else Do

            logger.info(f"Action dimension: {action_dim}, Observation feature dimension: {obs_feature_dim}")
            
            # 检查策略类型并相应地处理
            if hasattr(fast_policy, 'obs_encoder'):
                # 针对有obs_encoder的策略类型
                if fast_policy.obs_as_cond:
                    # 处理观察作为条件的情况
                    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                    nobs_features = fast_policy.obs_encoder(this_nobs)
                    # 重塑回 B, To, Do
                    cond = nobs_features.reshape(B, To, -1)
                    # 创建动作样本
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                else:
                    # 处理观察通过重绘条件的情况
                    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                    nobs_features = fast_policy.obs_encoder(this_nobs)
                    # 重塑回 B, To, Do
                    nobs_features = nobs_features.reshape(B, To, -1)
                    # 创建包含动作和观察特征的样本
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                    # 设置观察特征部分
                    sample[:,:To,action_dim:] = nobs_features
            else:
                # 针对DiffusionTransformerLowdimPolicy等没有obs_encoder的策略
                if hasattr(fast_policy, 'obs_as_cond') and fast_policy.obs_as_cond:
                    # 如果观察作为条件
                    cond = obs_dict['obs'][:,:To]
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                    if hasattr(fast_policy, 'pred_action_steps_only') and fast_policy.pred_action_steps_only:
                        # 如果只预测动作步骤
                        sample = torch.zeros(size=(B, fast_policy.n_action_steps, action_dim), device=device, dtype=dtype)
                else:
                    # 观察通过重绘条件处理
                    sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                    sample[:,:To,action_dim:] = obs_dict['obs'][:,:To]
            
            try:
                # 使用thop.profile计算FLOPs
                fast_policy.eval()  # 确保模型在评估模式
                macs, params = profile(fast_policy.model, inputs=(sample, timestep, cond), verbose=False)
                
                # MACs (乘加操作) 通常被视为FLOPs的两倍
                flops_value = macs * 2
                
                logger.info(f"模型参数数量: {params/1e6:.2f} M")
                logger.info(f"模型MACs: {macs/1e9:.4f} G")
                logger.info(f"模型FLOPs: {flops_value/1e9:.4f} G")
            except Exception as e:
                logger.error(f"计算FLOPs时出错: {e}")
                flops_value = 0.0
                import traceback
                traceback.print_exc()
    except Exception as e:
        logger.error(f"计算FLOPs时出错: {e}")
        flops_value = 0.0
        import traceback
        traceback.print_exc()

    
    # 性能测试
    logger.info(f"\n=== 性能测试 (动作帧/推理: {actions_per_inference}) ===")
    
    # 固定测试次数
    num_trials = 10
    
    # 测试原始策略
    logger.info("\n----- 原始策略 -----")
    original_durations = []
    
    for i in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            original_policy.predict_action(obs_dict) #原始策略预测：原始流程
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        original_durations.append(duration)
        logger.info(f"运行 {i+1}/{num_trials}: {duration:.4f} 秒")
    
    # 计算原始策略统计数据
    avg_original = np.mean(original_durations)
    frequency_original = actions_per_inference / avg_original
    logger.info(f"平均时间: {avg_original:.4f} 秒")
    logger.info(f"动作频率: {frequency_original:.2f} 动作/秒")
    
    # 测试缓存策略
    logger.info("\n----- 缓存策略 -----")
    fast_durations = []
    
    for i in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            fast_policy.predict_action(obs_dict)  #缓存策略预测：BU算法+缓存机制
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        fast_durations.append(duration)
        logger.info(f"运行 {i+1}/{num_trials}: {duration:.4f} 秒")
    
    # 计算缓存策略统计数据
    avg_fast = np.mean(fast_durations)
    frequency_fast = actions_per_inference / avg_fast
    speedup_time = avg_original / avg_fast
    
    logger.info(f"平均时间: {avg_fast:.4f} 秒")
    logger.info(f"动作频率: {frequency_fast:.2f} 动作/秒")
    logger.info(f"加速比: {speedup_time:.2f}x")
    
    # 保存结果
    cache_config = {
        "device": str(device),  # 将device转换为字符串
        "mode": cache_mode,
        "actions_per_inference": int(actions_per_inference),
        "num_bu_blocks": num_bu_blocks
    }
    
    if cache_mode == 'optimal':
        cache_config.update({
            "optimal_steps_dir": optimal_steps_dir,
            "metric": metric,
            "num_caches": num_caches,
            "multistep_method": multistep_method,
            "max_queue_length": max_queue_length,
            "precomputed_derivatives": precomputed_derivatives_path is not None
        })
        if precomputed_derivatives_path:
            cache_config["precomputed_derivatives_path"] = precomputed_derivatives_path
    else:  # threshold
        cache_config.update({
            "cache_threshold": cache_threshold,
        })
    
    # 保存benchmark结果
    benchmark_results = {
        **cache_config,
        "original": {
            "avg_time": float(avg_original),
            "frequency": float(frequency_original),
        },
        "fast": {
            "avg_time": float(avg_fast),
            "frequency": float(frequency_fast),
        },
        "speedup": float(speedup_time),
        "flops": float(flops_value),
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"基准测试结果已保存到 {output_dir}/benchmark_results.json")
    
    # 创建环境运行器
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    if skip_video:
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        logger.info("跳过视频渲染")
    
    # 创建环境运行器
    try:
        env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    except Exception as e:
        logger.error(f"实例化环境运行器时出错: {e}")
        try:
            # 如果失败，可能是因为参数不匹配，尝试检查KitchenLowdimRunner在初始化时需要的参数
            import inspect
            from diffusion_policy.env_runner.kitchen_lowdim_runner import KitchenLowdimRunner
            from diffusion_policy.env_runner.block_push_lowdim_runner import BlockPushLowdimRunner
            
            runner_class = None
            if "KitchenLowdimRunner" in str(env_runner_cfg):
                runner_class = KitchenLowdimRunner
                logger.info("检测到KitchenLowdimRunner")
            elif "BlockPushLowdimRunner" in str(env_runner_cfg):
                runner_class = BlockPushLowdimRunner
                logger.info("检测到BlockPushLowdimRunner")
            
            if runner_class:
                sig = inspect.signature(runner_class.__init__)
                logger.info(f"Runner初始化参数: {list(sig.parameters.keys())}")
                
                # 调整env_runner_cfg以匹配参数
                required_params = set(sig.parameters.keys()) - {'self'}
                for param in list(env_runner_cfg.keys()):
                    if param not in required_params and param != '_target_':
                        logger.info(f"移除不需要的参数: {param}")
                        del env_runner_cfg[param]
                
                # 确保output_dir包含在参数中
                env_runner_cfg['output_dir'] = output_dir
                
                # 重新尝试实例化
                env_runner = hydra.utils.instantiate(env_runner_cfg)
            else:
                raise e
        except Exception as inner_e:
            logger.error(f"尝试修复环境运行器实例化失败: {inner_e}")
            raise
    
    # 使用缓存策略评估
    logger.info("\n使用缓存策略运行环境评估...")
    fast_runner_log = env_runner.run(fast_policy)
    
    # 只保留test_mean_score
    test_mean_score = fast_runner_log.get('test/mean_score', 0.0)
    
    # 保存评估结果
    eval_results = {
        "mean_score": float(test_mean_score),
        "speedup": float(speedup_time),
        "flops": float(flops_value),
        "cache_mode": cache_mode,
        "num_caches": num_caches if cache_mode in ['optimal', 'fix', 'random'] else None,
        "num_bu_blocks": num_bu_blocks
    }
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"评估结果已保存到 {output_dir}/eval_results.json")
    logger.info(f"测试成功率: {test_mean_score:.4f}, 加速比: {speedup_time:.2f}x")
    if flops_value > 0:
        logger.info(f"FLOPs: {flops_value/1e9:.4f} GFLOPs")
    
    # 保存详细日志
    json_log = {k: v._path if hasattr(v, '_path') else v for k, v in fast_runner_log.items()}
    out_path = os.path.join(output_dir, 'fast_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f"详细评估日志已保存到 {out_path}")

if __name__ == "__main__":
    main()
