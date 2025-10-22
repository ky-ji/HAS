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
        使用拉格朗日插值公式预测当前特征
        
        公式：x(t) = (jk/(i-j)(i-k))·Xi + (ik/(j-i)(j-k))·Xj + (ij/(k-i)(k-j))·Xk
        
        其中：
        - i, j, k 是历史时间步
        - Xi, Xj, Xk 是对应的历史特征
        - t 是当前时间步
        
        策略：
        - 1个历史点：直接返回
        - 2个历史点：线性插值（一阶拉格朗日）
        - 3+个历史点：二次插值（二阶拉格朗日，使用最近3个点）
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
        t = current_timestep
        
        if num_history == 2:
            # ===== 线性插值（一阶拉格朗日）=====
            # x(t) = [(t-j)/(i-j)]·Xi + [(t-i)/(j-i)]·Xj
            i, j = timesteps[0], timesteps[1]
            Xi, Xj = features_tensor[0], features_tensor[1]
            
            if i == j:  # 避免除零
                return Xj
            
            # 拉格朗日一阶插值
            coeff_i = (t - j) / (i - j)
            coeff_j = (t - i) / (j - i)
            
            result = coeff_i * Xi + coeff_j * Xj
            
        else:
            # ===== 二次插值（二阶拉格朗日，使用最近3个点）=====
            # x(t) = (jk/(i-j)(i-k))·Xi + (ik/(j-i)(j-k))·Xj + (ij/(k-i)(k-j))·Xk
            
            # 使用最近的3个历史点
            i, j, k = timesteps[-3], timesteps[-2], timesteps[-1]
            Xi, Xj, Xk = features_tensor[-3], features_tensor[-2], features_tensor[-1]
            
            # 检查时间步是否不同（避免除零）
            if i == j or i == k or j == k:
                # 如果有重复时间步，退化为线性插值或直接返回
                return Xk
            
            # 计算拉格朗日插值系数
            # 注意：图片中的公式是 jk/(i-j)(i-k)，这里 jk 表示 j*k
            coeff_i = (j * k) / ((i - j) * (i - k))
            coeff_j = (i * k) / ((j - i) * (j - k))
            coeff_k = (i * j) / ((k - i) * (k - j))
            
            # 拉格朗日二次插值
            result = coeff_i * Xi + coeff_j * Xj + coeff_k * Xk
        
        return result
    
    def has(self) -> bool:
        return len(self.hash_table) > 0
    
    def clear(self):
        self.hash_table.clear()


