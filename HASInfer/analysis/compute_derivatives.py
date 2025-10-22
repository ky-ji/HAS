"""
离线计算激活的一阶导数（基于BU block selection的最优步骤）

功能：
1. 加载激活数据
2. 加载每个block的最优更新步骤（从optimal_steps目录）
3. 应用BU算法（与diffusion_hash_wrapper.py保持一致）
4. 只为需要的timesteps计算一阶导数
5. 保存导数数据供线性多步法使用

使用方法示例（修改相应文件路径）：
python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/kitchen/original/activations.pkl \
    --optimal_steps_dir assets/kitchen/original/optimal_steps/cosine \
    --output_dir assets/kitchen/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/lift_ph/original/activations.pkl \
    --optimal_steps_dir assets/lift_ph/original/optimal_steps/cosine \
    --output_dir assets/lift_ph/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/can_ph/original/activations.pkl \
    --optimal_steps_dir assets/can_ph/original/optimal_steps/cosine \
    --output_dir assets/can_ph/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/square_ph/original/activations.pkl \
    --optimal_steps_dir assets/square_ph/original/optimal_steps/cosine \
    --output_dir assets/square_ph/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/transport_ph/original/activations.pkl \
    --optimal_steps_dir assets/transport_ph/original/optimal_steps/cosine \
    --output_dir assets/transport_ph/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/tool_hang_ph/original/activations.pkl \
    --optimal_steps_dir assets/tool_hang_ph/original/optimal_steps/cosine \
    --output_dir assets/tool_hang_ph/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/block_pushing/original/activations.pkl \
    --optimal_steps_dir assets/block_pushing/original/optimal_steps/cosine \
    --output_dir assets/block_pushing/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

python -m HASInfer/analysis/compute_derivatives.py \
    --activations_path assets/pusht/original/activations.pkl \
    --optimal_steps_dir assets/pusht/original/optimal_steps/cosine \
    --output_dir assets/pusht/original/derivatives \
    --num_caches 7 \
    --metric cosine \
    --method five_point_stencil \
    --num_bu_blocks 5

"""
import time
import sys
import os
import argparse
import pickle
import torch
from pathlib import Path
import logging  
from tqdm import tqdm
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_derivative_central_difference(acts_list):   
    """
    中心差分法计算导数（最精确）
    
    f'(t) ≈ (f(t+1) - f(t-1)) / (2·Δt)
    
    Args:
        acts_list: List of tensors [act_t0, act_t1, ..., act_tn]
    
    Returns:
        derivs_list: List of derivative tensors
    """
    n = len(acts_list)
    derivs = []
    
    for t in range(n):
        if t == 0:
            # 前向差分（边界）
            deriv = acts_list[1] - acts_list[0]
        elif t == n - 1:
            # 后向差分（边界）
            deriv = acts_list[t] - acts_list[t-1]
        else:
            # 中心差分（内部点）
            deriv = (acts_list[t+1] - acts_list[t-1]) / 2.0
        
        derivs.append(deriv)
    
    return derivs


def compute_derivative_forward_difference(acts_list):
    """
    前向差分法计算导数
    
    f'(t) ≈ (f(t+1) - f(t)) / Δt
    """
    n = len(acts_list)
    derivs = []
    
    for t in range(n):
        if t < n - 1:
            deriv = acts_list[t+1] - acts_list[t]
        else:
            # 最后一个点使用后向差分
            deriv = acts_list[t] - acts_list[t-1]
        
        derivs.append(deriv)
    
    return derivs


def compute_derivative_backward_difference(acts_list):
    """
    后向差分法计算导数
    
    f'(t) ≈ (f(t) - f(t-1)) / Δt
    """
    n = len(acts_list)
    derivs = []
    
    for t in range(n):
        if t > 0:
            deriv = acts_list[t] - acts_list[t-1]
        else:
            # 第一个点使用前向差分
            deriv = acts_list[1] - acts_list[0]
        
        derivs.append(deriv)
    
    return derivs


def compute_derivative_five_point_stencil(acts_list):  #0.000050s
    """
    五点模板法（更高精度）
    
    f'(t) ≈ (-f(t+2) + 8f(t+1) - 8f(t-1) + f(t-2)) / (12·Δt)
    
    精度：O(h^4)
    """
    n = len(acts_list)
    derivs = []
    
    for t in range(n):
        if t < 2:
            # 前向差分（边界）
            deriv = acts_list[min(t+1, n-1)] - acts_list[t]
        elif t >= n - 2:
            # 后向差分（边界）
            deriv = acts_list[t] - acts_list[t-1]
        else:
            # 五点模板（内部点）
            deriv = (
                -acts_list[t+2] + 
                8.0 * acts_list[t+1] - 
                8.0 * acts_list[t-1] + 
                acts_list[t-2]
            ) / 12.0
        
        derivs.append(deriv)
    
    return derivs


def compute_derivative_bdf(acts_list):
    """
    BDF (Backward Differentiation Formula) 导数计算
    
    BDF方法特点：
    - 隐式方法，A-稳定性极好
    - 特别适合扩散模型的刚性问题
    - 对数值不稳定性有很强的抑制作用
    
    BDF公式（2阶）：
    f'(t) ≈ (3f(t) - 4f(t-1) + f(t-2)) / (2·Δt)
    
    BDF公式（3阶）：
    f'(t) ≈ (11f(t) - 18f(t-1) + 9f(t-2) - 2f(t-3)) / (6·Δt)
    """
    n = len(acts_list)
    derivs = []
    
    for t in range(n):
        if t == 0:
            # 第一个点：前向差分
            deriv = acts_list[1] - acts_list[0]
        elif t == 1:
            # 第二个点：后向差分（1阶BDF）
            deriv = acts_list[t] - acts_list[t-1]
        elif t == 2:
            # 第三个点：2阶BDF
            # f'(t) = (3f(t) - 4f(t-1) + f(t-2)) / 2
            deriv = (3.0 * acts_list[t] - 4.0 * acts_list[t-1] + acts_list[t-2]) / 2.0
        else:
            # 其余点：3阶BDF（最稳定）
            # f'(t) = (11f(t) - 18f(t-1) + 9f(t-2) - 2f(t-3)) / 6
            deriv = (
                11.0 * acts_list[t] - 
                18.0 * acts_list[t-1] + 
                9.0 * acts_list[t-2] - 
                2.0 * acts_list[t-3]
            ) / 6.0
        
        derivs.append(deriv)
    
    return derivs


def compute_derivative_adams_bashforth(acts_list):
    """
    Adams-Bashforth 导数计算（可用于验证和对比）
    
    AB方法特点：
    - 显式方法，计算快速
    - 适合平滑函数
    
    AB公式（3阶）：
    f'(t) ≈ (23f(t) - 16f(t-1) + 5f(t-2)) / 12
    """
    n = len(acts_list)
    derivs = []
    
    for t in range(n):
        if t == 0:
            deriv = acts_list[1] - acts_list[0]
        elif t == 1:
            deriv = acts_list[t] - acts_list[t-1]
        elif t == 2:
            # 2阶AB
            deriv = (3.0 * acts_list[t] - acts_list[t-1]) / 2.0
        else:
            # 3阶AB
            deriv = (
                23.0 * acts_list[t] - 
                16.0 * acts_list[t-1] + 
                5.0 * acts_list[t-2]
            ) / 12.0
        
        derivs.append(deriv)
    
    return derivs


def compute_derivatives_for_block(acts_list, update_steps, method='central_difference'):
    """
    为单个block的指定步骤计算导数
    
    注意：update_steps 应该是经过BU算法补充后的完整步骤列表，
    与 diffusion_hash_wrapper_multistep.py 中创建hash表的块的步骤完全一致
    
    Args:
        acts_list: 该block在所有timestep的激活列表 [act_t0, act_t1, ...]
        update_steps: 该block的最优更新步骤列表（BU算法处理后，需要计算导数的步骤）
        method: 差分方法
            - 'bdf': BDF方法
            - 'central_difference': 中心差分（精确）
            - 'adams_bashforth': AB方法
            - 'forward_difference': 前向差分
            - 'backward_difference': 后向差分
            - 'five_point_stencil': 五点模板（高精度）
    
    Returns:
        derivs_dict: {timestep: deriv_tensor}，只包含update_steps中的timestep
    """
    method_functions = {
        'central_difference': compute_derivative_central_difference,
        'forward_difference': compute_derivative_forward_difference,
        'backward_difference': compute_derivative_backward_difference,
        'five_point_stencil': compute_derivative_five_point_stencil,
        'bdf': compute_derivative_bdf,
        'adams_bashforth': compute_derivative_adams_bashforth,
    }
    
    if method not in method_functions:
        raise ValueError(f"Unknown method: {method}")
    
    # 首先为所有步骤计算导数（因为差分需要相邻点）
    compute_func = method_functions[method]
    all_derivs = compute_func(acts_list)
    
    # 只保留update_steps中的导数
    derivs_dict = {}
    update_steps_set = set(update_steps) if update_steps else set()
    
    for t in update_steps_set:
        if t < len(all_derivs):
            derivs_dict[t] = all_derivs[t]
    
    return derivs_dict


def identify_block_from_module_name(module_name):
    """
    从module名称识别block
    
    decoder.layers.0.dropout1 -> decoder.layers.0_sa_block
    decoder.layers.0.dropout2 -> decoder.layers.0_mha_block
    decoder.layers.0.dropout3 -> decoder.layers.0_ff_block
    """
    pattern = re.compile(r'decoder\.layers\.(\d+)\.dropout(\d+)')
    match = pattern.match(module_name)
    
    if match:
        layer_num = match.group(1)
        dropout_num = match.group(2)
        
        block_types = {
            '1': 'sa_block',
            '2': 'mha_block', 
            '3': 'ff_block'
        }
        
        if dropout_num in block_types:
            block_key = f"decoder.layers.{layer_num}_{block_types[dropout_num]}"
            return block_key
    
    return None


def organize_activations_by_block(activations):
    """
    将激活按block组织
    
    Returns:
        {block_key: [act_t0, act_t1, ...]}
    """
    block_activations = defaultdict(list)
    
    for module_name, acts_list in activations.items():
        block_key = identify_block_from_module_name(module_name)
        if block_key:
            block_activations[block_key] = acts_list
    
    return block_activations


def compute_derivatives_for_activations(activations, block_steps, method='central_difference'):
    """
    为所有 block 计算导数（基于optimal steps）
    
    Args:
        activations: {block_key: [act_t0, act_t1, ..., act_tn]}
        block_steps: {block_key: [step1, step2, ...]} BU算法处理后的步骤
        method: 差分方法
    
    Returns:
        derivatives: {block_key: {timestep: deriv_tensor}}
    """
    derivatives = {}
    
    logger.info(f"计算导数，方法: {method}")
    
    for block_key, acts_list in tqdm(activations.items(), desc="计算导数"):
        # 获取该block的更新步骤
        update_steps = block_steps.get(block_key, [])
        
        if not update_steps:
            logger.warning(f"  {block_key}: 没有更新步骤，跳过")
            continue
        
        # 计算该block的导数
        derivs_dict = compute_derivatives_for_block(acts_list, update_steps, method)
        derivatives[block_key] = derivs_dict
        
        logger.info(f"  {block_key}: 为 {len(derivs_dict)} 个timesteps计算了导数")
    
    return derivatives


def verify_derivatives(activations, derivatives, sample_blocks=3):
    """
    验证导数计算的质量
    
    通过重新计算数值导数并对比
    """
    logger.info("验证导数质量...")
    
    block_keys = list(activations.keys())[:sample_blocks]
    
    for block_key in block_keys:
        acts = activations[block_key]
        derivs = derivatives[block_key]
        
        # 选择几个测试点
        n = len(acts)
        test_points = [n//4, n//2, 3*n//4]
        
        logger.info(f"\n块: {block_key}")
        for t in test_points:
            if 0 < t < n - 1:
                # 数值导数（中心差分）
                numerical_deriv = (acts[t+1] - acts[t-1]) / 2.0
                stored_deriv = derivs[t]
                
                # 计算相对误差
                diff = numerical_deriv - stored_deriv
                relative_error = torch.norm(diff) / (torch.norm(numerical_deriv) + 1e-10)
                
                logger.info(f"  t={t}: 相对误差 = {relative_error:.6e}")


def save_derivatives(derivatives, output_path):
    """保存导数数据"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(derivatives, f)
    
    logger.info(f"导数已保存到: {output_path}")


def load_activations(activations_path):
    """加载激活数据"""
    logger.info(f"加载激活数据: {activations_path}")
    with open(activations_path, 'rb') as f:
        activations = pickle.load(f)
    logger.info(f"加载了 {len(activations)} 个模块的激活")
    return activations


def load_optimal_steps(optimal_steps_dir, num_caches, metric='cosine'):
    """
    加载每个block的最优更新步骤（与precompute_cache_weights.py保持一致）
    
    Args:
        optimal_steps_dir: 最优步骤目录
        num_caches: 缓存次数
        metric: 相似度指标
        
    Returns:
        {block_key: [step1, step2, ...]}
    """
    optimal_steps_dir = Path(optimal_steps_dir)
    block_steps = {}
    
    # 遍历所有dropout模块的步骤文件
    for steps_file in optimal_steps_dir.rglob(f'optimal_steps_*_{num_caches}_{metric}.pkl'):
        if 'dropout' in steps_file.parent.name:
            module_name = steps_file.parent.name
            block_key = identify_block_from_module_name(module_name)
            
            if block_key:
                with open(steps_file, 'rb') as f:
                    steps = pickle.load(f)
                block_steps[block_key] = steps
                logger.info(f"加载 {block_key}: {len(steps)} 个最优步骤")
    
    return block_steps


def apply_bu_algorithm(block_steps, bu_blocks):
    """
    应用BU算法补充步骤（与diffusion_hash_wrapper.py中的逻辑一致）
    """
    def get_layer_idx(block_key):
        match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
        if match:
            return int(match.group(1))
        return -1
    
    def get_block_type_priority(block_key):
        match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
        if match:
            block_type = match.group(1)
            priorities = {'sa_block': 0, 'mha_block': 1, 'ff_block': 2}
            return priorities.get(block_type, 10)
        return 10
    
    def get_block_type(block_key):
        match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
        if match:
            return match.group(1)
        return ""
    
    # BU传播：从列表中更深层到更浅层
    if len(bu_blocks) > 1:
        sorted_bu_blocks = sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x)))
        logger.info(f"BU: 排序后的块: {sorted_bu_blocks}")
        
        # 找出所有FFN块并按层索引排序
        all_ffn_blocks = []
        for block_key in block_steps.keys():
            if get_block_type(block_key) == 'ff_block':
                all_ffn_blocks.append(block_key)
        
        sorted_all_ffn_blocks = sorted(all_ffn_blocks, key=lambda x: get_layer_idx(x))
        
        # 为bu_blocks中的每个块从后层FFN Block获取steps
        for block_key in sorted_bu_blocks:
            block_layer_idx = get_layer_idx(block_key)
            
            deeper_ffn_blocks = []
            for ffn_block in sorted_all_ffn_blocks:
                ffn_layer_idx = get_layer_idx(ffn_block)
                if ffn_layer_idx >= block_layer_idx:
                    deeper_ffn_blocks.append(ffn_block)
            
            if deeper_ffn_blocks:
                all_deeper_steps = set()
                for deeper_ffn_block in deeper_ffn_blocks:
                    if deeper_ffn_block in block_steps:
                        all_deeper_steps.update(block_steps[deeper_ffn_block])
                
                if block_key in block_steps and all_deeper_steps:
                    current_steps = set(block_steps[block_key])
                    missing_steps = all_deeper_steps - current_steps
                    
                    if missing_steps:
                        updated_steps = sorted(list(current_steps.union(missing_steps)))
                        block_steps[block_key] = updated_steps
                        logger.info(f"BU: 为块 {block_key} 补充steps: {sorted(list(missing_steps))}")
    else:
        logger.info("BU算法：块数量不足，跳过")
    
    return block_steps


def load_bu_blocks(optimal_steps_dir, num_bu_blocks):
    """
    加载或计算误差最大的块（用于BU算法，逻辑与diffusion_hash_wrapper.py保持一致）
    """
    if num_bu_blocks <= 0:
        return []
    
    try:
        from diffusion_policy.activation_utils.bu_block_selection import analyze_block_errors
        
        # 从optimal_steps_dir推导基础目录路径
        if optimal_steps_dir:
            steps_path = Path(optimal_steps_dir)
            if "optimal_steps" in str(steps_path):
                base_path = steps_path.parent.parent
                output_base_dir = str(base_path.parent)
                task_name = base_path.name.split('/')[0]
                logger.info(f"从optimal_steps_dir推导出: output_base_dir={output_base_dir}, task_name={task_name}")
            else:
                raise ValueError("无法从optimal_steps_dir推导基础目录")
        else:
            raise ValueError("未提供optimal_steps_dir")
        
        bu_analysis_dir = f"{output_base_dir}/{task_name}/bu_block_selection"
        analysis_output_dir = Path(bu_analysis_dir)
        selected_path = analysis_output_dir / f'top_{num_bu_blocks}_error_blocks.pkl'
        
        if selected_path.exists():
            with open(selected_path, 'rb') as f:
                selected_blocks_dict = pickle.load(f)
                bu_blocks = list(selected_blocks_dict.keys())
                logger.info(f"从文件加载了 {len(bu_blocks)} 个误差最大的块用于BU算法: {bu_blocks}")
                return bu_blocks
        else:
            activations_path = Path(f"{output_base_dir}/{task_name}/activations.pkl")
            if activations_path.exists():
                logger.info(f"计算误差最大的块用于BU算法...")
                analysis_output_dir.mkdir(parents=True, exist_ok=True)
                selected_blocks = analyze_block_errors(str(activations_path), str(analysis_output_dir), num_bu_blocks)
                bu_blocks = [block for block, _ in selected_blocks]
                logger.info(f"计算得到 {len(bu_blocks)} 个误差最大的块用于BU算法: {bu_blocks}")
                return bu_blocks
            else:
                logger.warning(f"找不到激活文件: {activations_path}，无法应用BU算法")
                return []
    except Exception as e:
        logger.warning(f"加载或计算误差最大的块时出错: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='离线计算激活的一阶导数（基于BU block selection的最优步骤）')
    
    parser.add_argument('--activations_path', type=str, required=True,
                        help='激活数据文件路径 (.pkl)')
    parser.add_argument('--optimal_steps_dir', type=str, required=True,
                        help='最优步骤目录（包含optimal_steps文件）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='导数输出目录')
    parser.add_argument('--num_caches', type=int, default=30,
                        help='缓存次数（用于定位最优步骤文件）')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='相似度指标（用于定位最优步骤文件）')
    parser.add_argument('--num_bu_blocks', type=int, default=5,
                        help='应用BU算法的块数量，0表示不使用BU算法')
    parser.add_argument('--verify', action='store_true',
                        help='验证导数质量')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--method', type=str, default='five_point_stencil',
                        choices=['bdf', 'central_difference', 'adams_bashforth',
                        'forward_difference', 'backward_difference', 
                        'five_point_stencil'],
                        help='差分方法')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载激活
    activations = load_activations(args.activations_path)
    
    # 按block组织激活
    block_activations = organize_activations_by_block(activations)
    logger.info(f"找到 {len(block_activations)} 个block")
    
    # 加载最优步骤
    logger.info(f"从 {args.optimal_steps_dir} 加载最优步骤...")
    block_optimal_steps = load_optimal_steps(args.optimal_steps_dir, args.num_caches, args.metric)
    logger.info(f"成功加载 {len(block_optimal_steps)} 个block的最优步骤")
    
    # 应用BU算法（如果启用）
    if args.num_bu_blocks > 0:
        logger.info(f"应用BU算法，处理 {args.num_bu_blocks} 个误差最大的块...")
        bu_blocks = load_bu_blocks(args.optimal_steps_dir, args.num_bu_blocks)
        if bu_blocks:
            block_optimal_steps = apply_bu_algorithm(block_optimal_steps, bu_blocks)
            logger.info("BU算法应用完成")
        else:
            logger.warning("未能加载BU块，跳过BU算法")
    else:
        logger.info("BU算法已禁用（num_bu_blocks=0）")
    
    # 为每个block计算导数
    derivatives = compute_derivatives_for_activations(block_activations, block_optimal_steps, args.method)
    
    # 验证（如果需要）
    if args.verify:
        verify_derivatives(block_activations, derivatives)
    
    # 保存导数
    output_path = Path(args.output_dir) / f'derivatives_{args.method}.pkl'
    save_derivatives(derivatives, output_path)
    
    # 保存元信息
    meta_info = {
        'method': args.method,
        'num_blocks': len(derivatives),
        'blocks': list(derivatives.keys()),
        'activations_path': args.activations_path,
        'optimal_steps_dir': args.optimal_steps_dir,
        'num_caches': args.num_caches,
        'metric': args.metric,
        'num_bu_blocks': args.num_bu_blocks,
        'bu_algorithm_applied': args.num_bu_blocks > 0
    }
    
    meta_path = Path(args.output_dir) / f'derivatives_{args.method}_meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_info, f)
    logger.info(f"元信息已保存到: {meta_path}")
    
    # 打印统计信息
    logger.info("\n" + "="*60)
    logger.info("导数计算完成")
    logger.info("="*60)
    logger.info(f"方法: {args.method}")
    logger.info(f"处理了 {len(derivatives)} 个blocks")
    logger.info(f"最优步骤目录: {args.optimal_steps_dir}")
    logger.info(f"缓存次数: {args.num_caches}")
    logger.info(f"BU算法: {'已启用' if args.num_bu_blocks > 0 else '已禁用'} (num_bu_blocks={args.num_bu_blocks})")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

