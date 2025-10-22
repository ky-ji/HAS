"""
线性多步法缓存实现
支持 Adams-Bashforth-Moulton (AB) 预测方法

核心思想：
1. 离线阶段：计算历史导数（一阶导数）
2. 在线阶段：
   - Predict: 使用历史值+历史导数（AB方法）
   - 加权：多个历史特征预测值加权
"""
import time
import torch
import torch.nn.functional as F
from collections import deque
import math


class GridBasedHashTable_AdamsBashforth:
    """
    纯 Adams-Bashforth 方法（仅预测）
    """
    
    def __init__(self, max_queue_length=3, precomputed_derivatives=None, device=None):
        """
        Args:
            max_queue_length (int): 保存的历史步数
            precomputed_derivatives (dict): 离线预计算的导数字典 {timestep: derivative_tensor}
                                            （已在wrapper中转移到目标设备）
            device (torch.device or str): 目标设备，如 'cuda:0', 'cpu' 等
        """
        self.max_queue_length = max_queue_length
        self.device = device if device is not None else torch.device('cpu')
        self.hash_table = deque(maxlen=max_queue_length)
        
        # AB系数（在初始化时转换为tensor并移到目标设备）
        ab_coeffs_raw = {
            1: [1.0],
            2: [3.0/2.0, -1.0/2.0],
            3: [23.0/12.0, -16.0/12.0, 5.0/12.0],
            4: [55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0],
        }
        
        # 一次性将所有AB系数转换为tensor并移到目标设备
        self.ab_coeffs = {}
        for order, coeffs in ab_coeffs_raw.items():
            self.ab_coeffs[order] = torch.tensor(coeffs, dtype=torch.float32, device=self.device)
        
        # 预计算导数已在wrapper中转移到目标设备，直接使用
        self.precomputed_derivatives = precomputed_derivatives
        
        # AB系数dtype缓存（避免重复转换dtype）
        # 格式: {(order, dtype): coeffs_tensor}
        self.ab_coeffs_cache = {}
        
        # 融合权重缓存（根据dtype创建）
        # 格式: {(num_history, dtype): weights_tensor}
        self.fusion_weights_cache = {}
    
    def append(self, feature, timestep):
        """
        保存特征和时间步（不保存导数）
        
        Args:
            feature (torch.Tensor): block 的输出特征
            timestep (int): 当前的扩散时间步
        
        说明：
            只保存 (timestep, feature)，导数在 query 时按需获取/计算
        """
        feature = feature.detach()
        self.hash_table.append((timestep, feature))
    
    def _get_derivatives(self, timesteps, features):
        """
        获取历史导数（优先使用预计算，否则使用有限差分）
        
        Args:
            timesteps: 历史时间步列表
            features: 历史特征列表
        
        Returns:
            derivatives: 导数列表
        
        说明：
            预计算导数已在初始化时转移到正确设备，无需再次转换
        """
        derivatives = []
        
        for i, t in enumerate(timesteps):
            # 优先使用预计算导数
            if self.precomputed_derivatives is not None and t in self.precomputed_derivatives:
                deriv = self.precomputed_derivatives[t]
                derivatives.append(deriv)
                #print("调用离线计算导数")
            # 如果没有预计算导数，使用有限差分估算
            elif i > 0:
                #print("在线计算导数")
                dt = t - timesteps[i-1]
                if dt != 0:
                    deriv = (features[i] - features[i-1]) / dt
                else:
                    deriv = torch.zeros_like(features[i])
                derivatives.append(deriv)
            else:
                #print("在线计算导数")
                # 第一个点无法计算导数，设为零
                deriv = torch.zeros_like(features[i])
                derivatives.append(deriv)
        
        return derivatives
    
    def query(self, current_timestep):
        """
        基于多个历史特征的加权AB预测
        
        策略：
        1. 对每个历史特征进行AB多步法预测
        2. 对所有预测结果进行加权融合
        3. 较新的历史点权重更大
        
        优化点：
        - 提前stack所有导数和特征
        - 减少循环中的重复操作
        - 缓存系数的dtype转换
        """
        if len(self.hash_table) == 0:
            return None
        if len(self.hash_table) == 1:
            return self.hash_table[0][1]
        
        num_history = len(self.hash_table)
        timesteps = [t for t, _ in self.hash_table]
        features = [f for _, f in self.hash_table]
        
        # 提前stack所有特征为tensor
        features_tensor = torch.stack(features)  # [num_history, *feature_shape]
        dtype = features_tensor.dtype
        
        # 获取所有历史导数
        derivatives = self._get_derivatives(timesteps, features)

        if len(derivatives) == 0:
            return features[-1]
        
        # 提前stack所有导数为tensor
        derivatives_tensor = torch.stack(derivatives)  # [num_history, *feature_shape]
        
        # 预分配predictions列表
        predictions = []
        
        # 对每个历史特征进行AB预测
        for i in range(num_history):
            dt_i = current_timestep - timesteps[i]
            
            if dt_i == 0:
                predictions.append(features_tensor[i])
                continue
            
            # 使用张量切片
            available_derivs = derivatives_tensor[:i+1]  # [i+1, *feature_shape]
            order = min(i+1, 4)
            
            # 获取AB系数（带dtype缓存）
            coeffs_cache_key = (order, dtype)
            if coeffs_cache_key not in self.ab_coeffs_cache:
                coeffs = self.ab_coeffs[order].to(dtype=dtype)
                self.ab_coeffs_cache[coeffs_cache_key] = coeffs
            else:
                coeffs = self.ab_coeffs_cache[coeffs_cache_key]
            
            # 取最近的order个导数并反转
            recent_derivs = available_derivs[-order:]  # [order, *feature_shape]
            recent_derivs = torch.flip(recent_derivs, dims=[0])  # 反转顺序
            
            # 批量计算：f_pred = f_i + dt_i * sum(coeffs * derivs)
            coeff_shape = [order] + [1] * (recent_derivs.ndim - 1)
            coeffs_expanded = coeffs.view(coeff_shape)  # [order, 1, 1, ...]
            
            # 批量乘法和求和
            deriv_contribution = (dt_i * coeffs_expanded * recent_derivs).sum(dim=0)
            
            f_pred_i = features_tensor[i] + deriv_contribution
            predictions.append(f_pred_i)

        # 对所有预测结果进行加权融合
        # 获取或创建缓存的权重tensor
        cache_key = (num_history, dtype)
        
        if cache_key not in self.fusion_weights_cache:
            # 第一次使用，创建并缓存
            if num_history == 2:
                weights = torch.tensor([0.3, 0.7], dtype=dtype, device=self.device)
            elif num_history == 3:
                weights = torch.tensor([0.2, 0.30, 0.5], dtype=dtype, device=self.device)
            elif num_history == 4:
                weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=self.device)
            else:
                weights = torch.linspace(0.1, 0.9, num_history, dtype=dtype, device=self.device)
                weights = weights / weights.sum()
            
            self.fusion_weights_cache[cache_key] = weights
        else:
            weights = self.fusion_weights_cache[cache_key]

        # 加权求和
        predictions_tensor = torch.stack(predictions)  # [num_history, *feature_shape]
        weight_shape = [num_history] + [1] * (predictions_tensor.ndim - 1)
        final_pred = (predictions_tensor * weights.view(weight_shape)).sum(dim=0)
        
        return final_pred
    
    def has(self) -> bool:
        return len(self.hash_table) > 0
    
    def clear(self):
        self.has_table.clear()


