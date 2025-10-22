"""
集成线性多步法的 diffusion_hash_wrapper（完整版）

使用线性多步法AB进行缓存加速
"""

#导入线性多步法类 
from .hash_util_multistep import GridBasedHashTable_AdamsBashforth     


import time
import torch
import types
import logging
import pickle
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch.nn as nn

# 导入必要的类
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

# 创建全局logger
logger = logging.getLogger(__name__)

class FastDiffusionPolicyMultistep:
    """
    使用线性多步法加速扩散策略
    
    支持的方法：
    - AB (Adams-Bashforth)
    """

    @staticmethod
    def apply_cache(policy,
                   optimal_steps_dir: str = None,
                   num_caches: int = 30,
                   metric: str = 'cosine',
                   cache_mode: str = None,
                   num_bu_blocks: int = 5,
                   precomputed_derivatives_path: str = None,
                   multistep_method: str = 'ab',
                   max_queue_length: int = 3,
                   device: str = None,
                   **kwargs):
        """
        应用线性多步法缓存加速
        
        Args:
            policy: 原始扩散策略
            optimal_steps_dir: 最优步骤目录
            num_caches: 缓存更新次数
            metric: 指标类型
            cache_mode: 'original' 或 'optimal'
            num_bu_blocks: BU算法块数量
            precomputed_derivatives_path: 预计算导数文件路径
            multistep_method: 线性多步法类型
                - 'ab': Adams-Bashforth 
            max_queue_length: 历史步数
            device: 目标设备（如 'cuda:0', 'cpu'），None则自动检测
        
        Returns:
            更新后的策略实例
        """
        # 自动检测设备（如果未指定）
        if device is None:
            # 从模型参数中获取设备
            if hasattr(policy, 'model'):
                device = next(policy.model.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        logger.info(f"使用设备: {device}")
        # 确定缓存模式
        if cache_mode is None:
            cache_mode = 'optimal' if optimal_steps_dir else 'original'

        # 如果是原始模式，直接返回
        if cache_mode == 'original':
            logger.info("使用原始模式，不应用缓存加速")
            return policy

        # 只支持optimal模式
        if cache_mode != 'optimal':
            logger.warning(f"不支持的缓存模式: {cache_mode}，将使用original模式")
            return policy

        assert hasattr(policy, 'model'), "策略必须有model属性"
        model = policy.model
        assert isinstance(model, TransformerForDiffusion), "模型必须是TransformerForDiffusion类型"
        num_inference_steps = policy.num_inference_steps

        # 加载并转移预计算导数到目标设备
        precomputed_derivatives = None
        if precomputed_derivatives_path and Path(precomputed_derivatives_path).exists():
            logger.info(f"加载预计算导数: {precomputed_derivatives_path}")
            with open(precomputed_derivatives_path, 'rb') as f:
                precomputed_derivatives_cpu = pickle.load(f)
            logger.info(f"成功加载 {len(precomputed_derivatives_cpu)} 个block的导数")
            
            # 一次性将所有预计算导数转移到目标设备
            logger.info(f"开始将所有预计算导数转移到设备: {device}")
            precomputed_derivatives = {}
            total_tensors = 0
            for block_key, block_derivs in precomputed_derivatives_cpu.items():
                precomputed_derivatives[block_key] = {}
                for timestep, deriv in block_derivs.items():
                    if isinstance(deriv, torch.Tensor):
                        precomputed_derivatives[block_key][timestep] = deriv.to(device)
                        total_tensors += 1
                    else:
                        precomputed_derivatives[block_key][timestep] = deriv
            logger.info(f"✓ 成功转移 {total_tensors} 个导数张量到设备: {device}")
        else:
            logger.info("未提供导数数据，将使用在线有限差分估算")

        # 创建缓存结构
        cache = {
            'mode': 'optimal',
            'metric': metric,
            'optimal_steps_dir': optimal_steps_dir,
            'num_caches': num_caches,
            'num_steps': num_inference_steps,
            'current_step': -1,
            'block_cache': {},
            'block_hash': {},
            'block_steps': {},
            'num_bu_blocks': num_bu_blocks,
            'bu_blocks': {},
            'precomputed_derivatives': precomputed_derivatives,
            'multistep_method': multistep_method,
            'max_queue_length': max_queue_length,
            'device': device,  
        }
        policy._cache = cache

        # 找到所有需要缓存的层
        cacheable_layers = FastDiffusionPolicyMultistep._find_cacheable_layers(model)
        policy._cacheable_layers = cacheable_layers
        logger.info(f"找到 {len(cacheable_layers)} 个需要缓存的Transformer层")

        # 加载每个块的最优步骤
        FastDiffusionPolicyMultistep._load_block_optimal_steps(
            cache, cacheable_layers, optimal_steps_dir, num_caches, metric
        )

        # 为每个层添加缓存功能
        for layer_name, layer in cacheable_layers:
            FastDiffusionPolicyMultistep._add_cache_to_block(
                layer=layer, layer_name=layer_name, cache=cache
            )

        # 保存原始forward方法
        original_forward = model.forward

        # 创建带缓存功能的forward方法
        def forward_with_cache(self, sample, timestep, cond=None, **kwargs):
            """使用缓存的forward方法"""
            cache = getattr(policy, '_cache', None)
            
            # 增加步数计数器
            cache['current_step'] += 1
            current_step = cache['current_step']

            # 为所有块设置当前步骤和缓存标志
            FastDiffusionPolicyMultistep._update_cache_flags(cache, current_step)

            # 执行前向传播
            output = original_forward(sample, timestep, cond, **kwargs)
            
            return output

        # 替换模型的forward方法
        model.forward = types.MethodType(forward_with_cache, model)

        # 添加重置缓存方法
        def reset_cache(self):
            """重置缓存"""
            if hasattr(self, '_cache'):
                self._cache['current_step'] = -1
                self._cache['block_cache'] = {}
                # 清空hash表中的激活
                if 'block_hash' in self._cache:
                    for block_key, hash_table in self._cache['block_hash'].items():
                        if hasattr(hash_table, 'clear'):
                            hash_table.clear()
                self._cache['bu_blocks'] = {}
            return self

        policy.reset_cache = types.MethodType(reset_cache, policy)

        # 保存原始的predict_action方法
        original_predict_action = policy.predict_action

        # 创建带自动重置缓存的predict_action方法
        def predict_action_with_auto_reset(self, *args, **kwargs):
            """在每次predict_action调用前自动重置缓存"""
            self.reset_cache()
            return original_predict_action(*args, **kwargs)

        # 替换策略的predict_action方法
        policy.predict_action = types.MethodType(predict_action_with_auto_reset, policy)

        # 日志输出
        logger.info(f"线性多步法({multistep_method.upper()})缓存加速已应用")
        logger.info(f"  步骤目录: {optimal_steps_dir}")
        logger.info(f"  缓存次数: {num_caches}")
        logger.info(f"  历史步数: {max_queue_length}")
        logger.info(f"  导数数据: {'已加载' if precomputed_derivatives else '在线计算'}")
        
        if num_bu_blocks > 0:
            logger.info(f"  BU算法: 启用，应用于误差最大的 {num_bu_blocks} 个块")
        else:
            logger.info(f"  BU算法: 禁用")

        return policy

    @staticmethod
    def _find_cacheable_layers(model) -> List[Tuple[str, Any]]:
        """找到模型中所有可缓存的Transformer层"""
        cacheable_layers = []

        # 处理decoder部分
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                if isinstance(layer, nn.TransformerDecoderLayer):
                    name = f'decoder.layers.{i}'
                    cacheable_layers.append((name, layer))

        # 如果没有找到常规的decoder层，尝试查找其他类型的Transformer层
        if not cacheable_layers:
            for name, module in model.named_modules():
                if isinstance(module, nn.TransformerDecoderLayer):
                    cacheable_layers.append((name, module))

        logger.info(f"共找到 {len(cacheable_layers)} 个可缓存的Transformer层")
        return cacheable_layers

    @staticmethod
    def _load_block_optimal_steps(cache: Dict, layers: List[Tuple[str, Any]],
                                 optimal_steps_dir: str, num_caches: int, metric: str):
        """加载每个块的最优步骤"""
        steps_dir = Path(optimal_steps_dir)
        assert steps_dir.exists(), f"最优步骤目录 {optimal_steps_dir} 不存在"

        block_steps = {}

        # 加载每个块的步骤
        for layer_name, layer in layers:
            dropout1_name = f"{layer_name}.dropout1"
            dropout2_name = f"{layer_name}.dropout2"
            dropout3_name = f"{layer_name}.dropout3"

            # sa_block使用dropout1的步骤
            sa_block_key = f"{layer_name}_sa_block"
            steps_file = steps_dir/dropout1_name/f'optimal_steps_{dropout1_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[sa_block_key] = steps
                    logger.info(f"为自注意力块 {sa_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {sa_block_key} 的步骤失败: {e}")

            # mha_block使用dropout2的步骤
            mha_block_key = f"{layer_name}_mha_block"
            steps_file = steps_dir/dropout2_name/f'optimal_steps_{dropout2_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[mha_block_key] = steps
                    logger.info(f"为多头注意力块 {mha_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {mha_block_key} 的步骤失败: {e}")

            # ff_block使用dropout3的步骤
            ff_block_key = f"{layer_name}_ff_block"
            steps_file = steps_dir/dropout3_name/f'optimal_steps_{dropout3_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[ff_block_key] = steps
                    logger.info(f"为前馈网络块 {ff_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {ff_block_key} 的步骤失败: {e}")

        # 应用BU算法
        if cache.get('num_bu_blocks', 0) > 0:
            logger.info("Applying BU (Bottom-Up) steps propagation.")
            
            bu_blocks = cache.get('bu_blocks', None)

            if len(bu_blocks) == 0:
                # 尝试从分析结果加载误差最大的块
                try:
                    from diffusion_policy.activation_utils.bu_block_selection import analyze_block_errors
                    
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
                    
                    num_blocks = cache.get('num_bu_blocks', 3)
                    bu_analysis_dir = f"{output_base_dir}/{task_name}/bu_block_selection"
                    analysis_output_dir = Path(bu_analysis_dir)
                    selected_path = analysis_output_dir / f'top_{num_blocks}_error_blocks.pkl'
                    
                    if selected_path.exists():
                        with open(selected_path, 'rb') as f:
                            selected_blocks_dict = pickle.load(f)
                            bu_blocks = list(selected_blocks_dict.keys())
                            logger.info(f"从文件加载了 {len(bu_blocks)} 个误差最大的块用于BU算法: {bu_blocks}")
                    else:
                        activations_path = Path(f"{output_base_dir}/{task_name}/activations.pkl")
                        if activations_path.exists():
                            logger.info(f"计算误差最大的块用于BU算法...")
                            analysis_output_dir.mkdir(parents=True, exist_ok=True)
                            selected_blocks = analyze_block_errors(str(activations_path), str(analysis_output_dir), num_blocks)
                            bu_blocks = [block for block, _ in selected_blocks]
                            logger.info(f"计算得到 {len(bu_blocks)} 个误差最大的块用于BU算法: {bu_blocks}")
                        else:
                            logger.warning(f"找不到激活文件: {activations_path}，无法应用BU算法")
                            bu_blocks = []
                except Exception as e:
                    logger.warning(f"加载或计算误差最大的块时出错: {e}")
                    bu_blocks = []
            
            # BU传播
            if len(bu_blocks) > 1:
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
                
                    # 创建线性多步法 hash 表
                    if 'block_hash' not in cache:
                        cache['block_hash'] = {}
                    
                    if block_key not in cache['block_hash']:
                        cache['block_hash'][block_key] = FastDiffusionPolicyMultistep._create_hash_table(
                            cache, block_key, device=cache.get('device')
                        )
                        logger.info(f"为块 {block_key} 创建 {cache['multistep_method'].upper()} hash表")
            
            logger.info(f"BU算法应用完成，处理了 {len(bu_blocks)} 个块")

        # 存储到缓存
        cache['block_steps'] = block_steps
        logger.info(f"共为 {len(block_steps)} 个计算块加载了最优步骤")

    @staticmethod
    def _create_hash_table(cache: Dict, block_key: str, device=None):
        """
        根据配置创建合适的 hash 表
        
        Args:
            cache: 缓存字典
            block_key: block 标识符
            device: 目标设备（如 'cuda:0', 'cpu'）
        
        Returns:
            hash_table: 对应的 hash 表实例
        
        说明：
            在初始化时为每个block加载对应的预计算导数并转移到目标设备
        """
        method = cache.get('multistep_method', 'abm')
        max_queue_length = cache.get('max_queue_length', 3)
        
        # 获取该block的预计算导数（已在apply_cache中一次性加载）
        block_derivatives = None
        if cache.get('precomputed_derivatives') and block_key in cache['precomputed_derivatives']:
            block_derivatives = cache['precomputed_derivatives'][block_key]
            logger.info(f"为块 {block_key} 加载预计算导数")
        

        if method == 'ab':
            # AB 纯预测（最快）
            hash_table = GridBasedHashTable_AdamsBashforth(
                max_queue_length=max_queue_length,
                precomputed_derivatives=block_derivatives,
                device=device
            )

        else:
            raise ValueError(f"未知的多步法: {method}")
        
        return hash_table


    @staticmethod
    def _update_cache_flags(cache: Dict, current_step: int):
        """更新所有块的缓存标志"""
        # 性能优化：使用字典预先存储所有steps
        if 'steps_lookup' not in cache:
            steps_lookup = {}
            for block_key, steps in cache['block_steps'].items():
                if not steps:
                    steps_lookup[block_key] = set()
                else:
                    steps_lookup[block_key] = set(steps)
                    # 添加步骤99-100作为强制缓存步骤
                    steps_lookup[block_key].add(99)
                    steps_lookup[block_key].add(100)
            cache['steps_lookup'] = steps_lookup

        # 使用查找表进行O(1)查找
        should_cache = {}
        for block_key, step_set in cache['steps_lookup'].items():
            if not step_set:
                should_cache[block_key] = False
            else:
                should_cache[block_key] = current_step in step_set

        cache['should_cache'] = should_cache

    @staticmethod
    def _add_cache_to_block(layer, layer_name: str, cache: Dict):
        """为Transformer层添加缓存功能"""
        # 保存原始forward方法
        original_forward = layer.forward

        # 只处理TransformerDecoderLayer
        assert isinstance(layer, nn.TransformerDecoderLayer), "只支持TransformerDecoderLayer"
        
        def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                              tgt_key_padding_mask=None, memory_key_padding_mask=None,
                              tgt_is_causal=None, memory_is_causal=False):

            # 缓存键
            sa_block_key = f"{layer_name}_sa_block"
            mha_block_key = f"{layer_name}_mha_block"
            ff_block_key = f"{layer_name}_ff_block"

            # 确定是否应该更新缓存
            block_should_cache = cache.get('should_cache', {})
            should_cache_sa = block_should_cache.get(sa_block_key, False)
            should_cache_mha = block_should_cache.get(mha_block_key, False)
            should_cache_ff = block_should_cache.get(ff_block_key, False)

            # 获取当前时间步
            current_step = cache.get('current_step', 0)
            
            # latent设置
            x = tgt

            def is_more_error_block(block_key):
                return block_key in cache['block_hash']

            has_sa_cache = sa_block_key in cache['block_cache'] and not should_cache_sa
            has_mha_cache = mha_block_key in cache['block_cache'] and not should_cache_mha
            has_ff_cache = ff_block_key in cache['block_cache'] and not should_cache_ff

            # 自注意力部分 
            if has_sa_cache:
                if is_more_error_block(sa_block_key):
                    sa_cache = cache['block_hash'][sa_block_key].query(current_step)
                else:
                    sa_cache = cache['block_cache'][sa_block_key]
                x = x + sa_cache
            else:
                sa_result = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                if is_more_error_block(sa_block_key):
                    cache['block_hash'][sa_block_key].append(sa_result.detach(), current_step)
                    cache['block_cache'][sa_block_key] = 1  #存在标志
                else:
                    cache['block_cache'][sa_block_key] = sa_result.detach()
                x = x + sa_result
            # 多头注意力部分 
            if has_mha_cache:
                if is_more_error_block(mha_block_key):
                    mha_cache = cache['block_hash'][mha_block_key].query(current_step)
                else:
                    mha_cache = cache['block_cache'][mha_block_key]
                x = x + mha_cache
            else:
                mha_result = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                if is_more_error_block(mha_block_key):
                    cache['block_hash'][mha_block_key].append(mha_result.detach(), current_step)
                    cache['block_cache'][mha_block_key] = 1  #存在标志
                else:
                    cache['block_cache'][mha_block_key] = mha_result.detach()
                x = x + mha_result

            # 前馈网络部分
            if has_ff_cache:
                if is_more_error_block(ff_block_key):
                    ff_cache = cache['block_hash'][ff_block_key].query(current_step)
                else:
                    ff_cache = cache['block_cache'][ff_block_key]
                x = x + ff_cache
            else:
                ff_result = self._ff_block(self.norm3(x))
                if is_more_error_block(ff_block_key):
                    cache['block_hash'][ff_block_key].append(ff_result.detach(), current_step)
                    cache['block_cache'][ff_block_key] = 1  #存在标志
                else:
                    cache['block_cache'][ff_block_key] = ff_result.detach()
                x = x + ff_result

            return x

        # 替换层的forward方法
        layer.forward = types.MethodType(forward_with_cache, layer)
