#!/bin/bash

# 计算BU算法后各类block的实际steps总数
# 此脚本用于分析在应用BU算法后，每个block实际运行的steps数量

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录（假设脚本在shs/get_steps/目录下）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"
echo "工作目录: $(pwd)"

# 默认参数
DEVICE="cuda:0"
NUM_CACHES="7"
METRIC="cosine"
DEBUG_MODE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_caches)
            NUM_CACHES="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --task)
            SINGLE_TASK="$2"
            shift 2
            ;;
        --num_bu_blocks)
            NUM_BU_BLOCKS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置默认的BU块数量
if [ -z "$NUM_BU_BLOCKS" ]; then
    NUM_BU_BLOCKS=5
fi

# 定义PH任务列表
declare -A PH_TASKS=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)

# 定义MH任务列表
declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)

# 定义低维任务列表
declare -A LOWDIM_TASKS=(
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

# 初始化任务类型FLOPS统计数组
declare -a PH_ORIGINAL_FLOPS=()
declare -a PH_PRACTICAL_FLOPS=()
declare -a MH_ORIGINAL_FLOPS=()
declare -a MH_PRACTICAL_FLOPS=()
declare -a LOWDIM_ORIGINAL_FLOPS=()
declare -a LOWDIM_PRACTICAL_FLOPS=()

# 处理单个任务的情况
if [ ! -z "$SINGLE_TASK" ]; then
    if [[ -v PH_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理PH任务: $SINGLE_TASK"
        declare -A PH_TASKS_FILTERED
        PH_TASKS_FILTERED["$SINGLE_TASK"]="${PH_TASKS[$SINGLE_TASK]}"
        PH_TASKS=()
        for key in "${!PH_TASKS_FILTERED[@]}"; do
            PH_TASKS["$key"]="${PH_TASKS_FILTERED[$key]}"
        done
        MH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v MH_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理MH任务: $SINGLE_TASK"
        declare -A MH_TASKS_FILTERED
        MH_TASKS_FILTERED["$SINGLE_TASK"]="${MH_TASKS[$SINGLE_TASK]}"
        MH_TASKS=()
        for key in "${!MH_TASKS_FILTERED[@]}"; do
            MH_TASKS["$key"]="${MH_TASKS_FILTERED[$key]}"
        done
        PH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v LOWDIM_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理低维任务: $SINGLE_TASK"
        declare -A LOWDIM_TASKS_FILTERED
        LOWDIM_TASKS_FILTERED["$SINGLE_TASK"]="${LOWDIM_TASKS[$SINGLE_TASK]}"
        LOWDIM_TASKS=()
        for key in "${!LOWDIM_TASKS_FILTERED[@]}"; do
            LOWDIM_TASKS["$key"]="${LOWDIM_TASKS_FILTERED[$key]}"
        done
        PH_TASKS=()
        MH_TASKS=()
    else
        echo "错误: 找不到任务 $SINGLE_TASK"
        exit 1
    fi
fi

# 显示任务信息
echo "===========================================" 
echo "开始计算BU算法后各类block实际steps总数"
echo "===========================================" 
echo "设备: $DEVICE"
echo "缓存数量: $NUM_CACHES"
echo "相似度指标: $METRIC"
echo "BU块数量: $NUM_BU_BLOCKS"
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式: 开启 (只打印任务信息，不执行命令)"
fi
echo "===========================================" 

# 创建结果输出目录
RESULT_DIR="results/practical_steps"
mkdir -p "$RESULT_DIR"

# 函数: 处理一个任务的实际steps统计
process_task() {
    local task_name=$1
    local checkpoint=$2
    local task_type=$3  # 新增：任务类型 (PH/MH/LOWDIM)
    
    echo "处理任务: $task_name (类型: $task_type)"
    echo "检查点: $checkpoint"
    
    # 确定步骤目录
    local steps_dir="assets/${task_name}/original/optimal_steps"
    if [ ! -d "$steps_dir" ]; then
        echo "错误: 步骤目录不存在: $steps_dir"
        return 1
    fi
    
    # 确定相似度度量子目录
    local metric_dir="${steps_dir}/${METRIC}"
    if [ ! -d "$metric_dir" ]; then
        echo "错误: 相似度度量目录不存在: $metric_dir"
        echo "可用的相似度度量: $(ls $steps_dir)"
        return 1
    fi
    
    # 确定BU块文件
    local bu_blocks_file="assets/${task_name}/original/bu_block_selection/top_${NUM_BU_BLOCKS}_error_blocks.pkl"
    if [ ! -f "$bu_blocks_file" ]; then
        echo "错误: BU块文件不存在: $bu_blocks_file"
        echo "可用的BU块文件: $(ls assets/${task_name}/original/bu_block_selection/)"
        return 1
    fi
    
    # 如果是调试模式，则只显示信息不执行
    if [ "$DEBUG_MODE" = true ]; then
        echo "调试模式: 跳过执行"
        return 0
    fi
    
    # 输出文件
    local output_file="${RESULT_DIR}/${task_name}_bu${NUM_BU_BLOCKS}_${METRIC}_caches${NUM_CACHES}_practical_steps.csv"
    
    # 传递变量到Python脚本
    # 注意：这里显式地传递bash变量到Python中
    output=$(python -c "
import os
import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 从bash传递的变量
NUM_CACHES = $NUM_CACHES
METRIC = \"$METRIC\"
TASK_TYPE = \"$task_type\"  # 任务类型

# 定义块解析函数，与diffusion_cache_wrapper.py中的完全一致
def get_layer_idx(block_key):
    match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
    if match:
        return int(match.group(1))
    
    # 兼容 decoder.layers.X.dropoutY 格式
    match = re.match(r'decoder\.layers\.(\d+)\.dropout\d+', block_key)
    if match:
        return int(match.group(1))
    
    return -1

def get_block_type_priority(block_key):
    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
    if match:
        block_type = match.group(1)
        priorities = {'sa_block': 0, 'mha_block': 1, 'ff_block': 2}
        return priorities.get(block_type, 10)  # 未知类型给高优先级
    
    # 兼容 decoder.layers.X.dropoutY 格式
    match = re.match(r'decoder\.layers\.\d+\.dropout(\d+)', block_key)
    if match:
        dropout_num = int(match.group(1))
        if dropout_num == 1:
            return 0  # sa_block
        elif dropout_num == 2:
            return 1  # mha_block
        elif dropout_num == 3:
            return 2  # ff_block
    
    return 10

def get_block_type(block_key):
    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
    if match:
        return match.group(1)
    
    # 兼容 decoder.layers.X.dropoutY 格式
    match = re.match(r'decoder\.layers\.\d+\.dropout(\d+)', block_key)
    if match:
        dropout_num = int(match.group(1))
        if dropout_num == 1:
            return 'sa_block'
        elif dropout_num == 2:
            return 'mha_block'
        elif dropout_num == 3:
            return 'ff_block'
    
    return ''

# 计算FLOPS的函数
def calculate_flops(ff_steps, sa_steps, mha_steps):
    
    base_flops = 0.4116
    ff_coeff = 0.01048
    sa_coeff = 0.005294
    mha_coeff = 0.003424
    
    return base_flops + (ff_coeff * ff_steps) + (sa_coeff * sa_steps) + (mha_coeff * mha_steps)

# 转换块名称格式 (从 dropout 格式转换为 block 格式)
def convert_to_block_format(block_key):
    match = re.match(r'decoder\.layers\.(\d+)\.dropout(\d+)', block_key)
    if match:
        layer_idx = match.group(1)
        dropout_num = int(match.group(2))
        
        if dropout_num == 1:
            return f'decoder.layers.{layer_idx}_sa_block'
        elif dropout_num == 2:
            return f'decoder.layers.{layer_idx}_mha_block'
        elif dropout_num == 3:
            return f'decoder.layers.{layer_idx}_ff_block'
    
    return block_key  # 如果已经是正确格式，则不变

# 加载BU块
bu_blocks_file = \"$bu_blocks_file\"
bu_blocks = []
try:
    with open(bu_blocks_file, 'rb') as f:
        bu_blocks_dict = pickle.load(f)
        bu_blocks = list(bu_blocks_dict.keys())
        # 确保BU块使用正确的格式
        bu_blocks = [convert_to_block_format(block) for block in bu_blocks]
    print(f'加载了 {len(bu_blocks)} 个BU块: {bu_blocks}')
except Exception as e:
    print(f'加载BU块失败: {e}')
    bu_blocks = []

# 初始化数据结构
steps_dir = Path(\"$metric_dir\")  # 已修改为使用相似度度量子目录
practical_steps = {}
block_info = {}

# 调试: 列出所有目录和文件
print(f'目录结构: {list(steps_dir.iterdir()) if steps_dir.exists() else "目录不存在"}')

# 遍历所有block目录，加载原始步骤
if not steps_dir.exists():
    print(f'错误: 步骤目录不存在: {steps_dir}')
    exit(1)
    
for block_dir in steps_dir.iterdir():
    if block_dir.is_dir():
        original_block_key = block_dir.name
        block_key = convert_to_block_format(original_block_key)
        block_type = get_block_type(block_key)
        layer_idx = get_layer_idx(block_key)
        
        print(f'处理目录: {original_block_key} -> 转换为块: {block_key}, 类型: {block_type}, 层索引: {layer_idx}')
        
        # 加载步骤文件
        steps_file = block_dir / f'optimal_steps_{original_block_key}_{NUM_CACHES}_{METRIC}.pkl'
        if steps_file.exists():
            try:
                with open(steps_file, 'rb') as f:
                    steps = pickle.load(f)
                if block_key not in practical_steps:
                    practical_steps[block_key] = steps
                    block_info[block_key] = {
                        'block_type': block_type,
                        'layer_idx': layer_idx,
                        'original_steps': steps,
                        'is_bu_block': block_key in bu_blocks,
                        'practical_steps': steps  # 初始值与原始steps相同
                    }
                    print(f'加载块 {block_key} 的步骤: {steps}')
            except Exception as e:
                print(f'加载步骤文件失败 {steps_file}: {e}')

# 如果没有块或BU块，直接返回
if not practical_steps:
    print('没有找到任何块，无法计算实际步骤')
    exit(1)
    
if len(bu_blocks) == 0:
    print('没有找到BU块，无法应用BU算法')
    exit(1)

# 打印所有加载的块信息
print(f'加载了 {len(practical_steps)} 个块的步骤:')
for block_key in practical_steps.keys():
    print(f'  - {block_key} (类型: {get_block_type(block_key)})')

# 应用BU算法逻辑，与diffusion_cache_wrapper.py中的完全一致
if len(bu_blocks) > 1:
    # 排序BU块
    sorted_bu_blocks = sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x)))
    print(f'BU: 排序后的块: {sorted_bu_blocks}')
    
    # 找出所有FFN块（无论是否在bu_blocks中）并按层索引排序
    all_ffn_blocks = []
    for block_key in practical_steps.keys():
        if get_block_type(block_key) == 'ff_block':
            all_ffn_blocks.append(block_key)
    
    sorted_all_ffn_blocks = sorted(all_ffn_blocks, key=lambda x: get_layer_idx(x))
    print(f'第一阶段，BU: 所有FFN块（按层索引排序）: {sorted_all_ffn_blocks}')
    
    # 为bu_blocks中的每个块寻找其后层的所有FFN Block
    for i, block_key in enumerate(sorted_bu_blocks):
        block_layer_idx = get_layer_idx(block_key)
        print(f'处理BU块: {block_key}, 层索引: {block_layer_idx}')
        
        # 收集所有更深层的FFN Block
        deeper_ffn_blocks = []
        for ffn_block in sorted_all_ffn_blocks:
            ffn_layer_idx = get_layer_idx(ffn_block)
            if ffn_layer_idx >= block_layer_idx:
                deeper_ffn_blocks.append(ffn_block)
        
        if deeper_ffn_blocks:
            print(f'BU: 为块 {block_key} (层 {block_layer_idx}) 找到后层FFN块: {deeper_ffn_blocks}')
            
            # 合并所有后层FFN Block的steps
            all_deeper_steps = set()
            for deeper_ffn_block in deeper_ffn_blocks:
                if deeper_ffn_block in practical_steps:
                    all_deeper_steps.update(practical_steps[deeper_ffn_block])
            
            # 确保当前块具有所有后层FFN Block的steps
            if block_key in block_info and all_deeper_steps:
                current_steps = set(block_info[block_key]['original_steps'])
                missing_steps = all_deeper_steps - current_steps
                
                if missing_steps:
                    updated_steps = sorted(list(current_steps.union(missing_steps)))
                    block_info[block_key]['practical_steps'] = updated_steps
                    practical_steps[block_key] = updated_steps
                    print(f'BU: 为块 {block_key} 补充后层FFN Block的steps: {sorted(list(missing_steps))}。新steps: {updated_steps}')
        else:
            print(f'没有找到比块 {block_key} 更深的FFN块')
else:
    print('BU: 参与传播的块不足（至少需要2个）。')

# 准备结果数据
results = []
for block_key, info in block_info.items():
    original_count = len(info['original_steps'])
    practical_count = len(info['practical_steps'])
    
    results.append({
        'block_key': block_key,
        'block_type': info['block_type'],
        'layer_idx': info['layer_idx'],
        'is_bu_block': info['is_bu_block'],
        'original_steps': original_count,
        'practical_steps': practical_count,
        'added_steps': practical_count - original_count,
        'increase_percent': (practical_count - original_count) / original_count * 100 if original_count > 0 else 0
    })

# 按块类型和层索引排序结果
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['block_type', 'layer_idx'])

# 计算每种类型的汇总信息
summary = results_df.groupby('block_type').agg({
    'original_steps': 'sum',
    'practical_steps': 'sum',
    'added_steps': 'sum',
    'increase_percent': 'mean'
}).reset_index()

# 添加总计行
total_row = {
    'block_type': 'total',
    'original_steps': results_df['original_steps'].sum(),
    'practical_steps': results_df['practical_steps'].sum(),
    'added_steps': results_df['added_steps'].sum(),
    'increase_percent': results_df['increase_percent'].mean()
}
summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

# 计算FLOPS
# 获取各类型块的步骤数量
ff_original_steps = summary.loc[summary['block_type'] == 'ff_block', 'original_steps'].values[0] if 'ff_block' in summary['block_type'].values else 0
sa_original_steps = summary.loc[summary['block_type'] == 'sa_block', 'original_steps'].values[0] if 'sa_block' in summary['block_type'].values else 0
mha_original_steps = summary.loc[summary['block_type'] == 'mha_block', 'original_steps'].values[0] if 'mha_block' in summary['block_type'].values else 0

ff_practical_steps = summary.loc[summary['block_type'] == 'ff_block', 'practical_steps'].values[0] if 'ff_block' in summary['block_type'].values else 0
sa_practical_steps = summary.loc[summary['block_type'] == 'sa_block', 'practical_steps'].values[0] if 'sa_block' in summary['block_type'].values else 0
mha_practical_steps = summary.loc[summary['block_type'] == 'mha_block', 'practical_steps'].values[0] if 'mha_block' in summary['block_type'].values else 0

# 计算原始和实际FLOPS
original_flops = calculate_flops(ff_original_steps, sa_original_steps, mha_original_steps)
practical_flops = calculate_flops(ff_practical_steps, sa_practical_steps, mha_practical_steps)

# 计算FLOPS增加比例
flops_increase = practical_flops - original_flops
flops_increase_percent = (flops_increase / original_flops) * 100 if original_flops > 0 else 0

# 打印任务名称和FLOPS
# 使用纯字符串而不是f-string嵌套变量
task_name = \"$task_name\"
print(\"\\n===== \" + TASK_TYPE + \"任务: \" + task_name + \" FLOPS计算结果 =====\")
print(f'原始FLOPS: {original_flops:.6f}')
print(f'实际FLOPS: {practical_flops:.6f}')
print(f'FLOPS增加: {flops_increase:.6f} ({flops_increase_percent:.2f}%)')
print('===========================\\n')

# 将FLOPS结果添加到汇总中
flops_row = {
    'block_type': 'flops',
    'original_steps': original_flops,
    'practical_steps': practical_flops,
    'added_steps': flops_increase,
    'increase_percent': flops_increase_percent
}
summary = pd.concat([summary, pd.DataFrame([flops_row])], ignore_index=True)

# 保存结果
output_file = \"$output_file\"
results_df.to_csv(output_file, index=False)
print(f'已保存详细结果到: {output_file}')

# 保存汇总信息
summary_file = output_file.replace('.csv', '_summary.csv')
summary.to_csv(summary_file, index=False)
print(f'已保存汇总结果到: {summary_file}')

# 打印汇总信息
print('\\n===== 汇总信息 =====')
print(summary)
print('===================\\n')

# 将FLOPS结果返回给bash
print(f'FLOPS_RESULTS:{original_flops}:{practical_flops}')
" 2>&1) || {
        echo "执行Python脚本失败"
        return 1
    }
    
    echo "$output"
    
    # 提取FLOPS结果，格式为"FLOPS_RESULTS:原始FLOPS:实际FLOPS"
    flops_line=$(echo "$output" | grep "FLOPS_RESULTS:" | tail -1)
    if [[ ! -z "$flops_line" ]]; then
        original_flops=$(echo "$flops_line" | cut -d':' -f2)
        practical_flops=$(echo "$flops_line" | cut -d':' -f3)
        
        # 根据任务类型存储FLOPS结果
        if [[ "$task_type" == "PH" ]]; then
            PH_ORIGINAL_FLOPS+=("$original_flops")
            PH_PRACTICAL_FLOPS+=("$practical_flops")
        elif [[ "$task_type" == "MH" ]]; then
            MH_ORIGINAL_FLOPS+=("$original_flops")
            MH_PRACTICAL_FLOPS+=("$practical_flops")
        elif [[ "$task_type" == "LOWDIM" ]]; then
            LOWDIM_ORIGINAL_FLOPS+=("$original_flops")
            LOWDIM_PRACTICAL_FLOPS+=("$practical_flops")
        fi
    fi
    
    echo "任务 ${task_name} 计算完成"
    echo "-------------------------------------------"
}

# 计算数组平均值的函数
calculate_average() {
    local sum=0
    local count=0
    
    for value in "$@"; do
        sum=$(echo "$sum + $value" | bc -l)
        count=$((count + 1))
    done
    
    if [[ $count -eq 0 ]]; then
        echo "0"
    else
        echo "scale=6; $sum / $count" | bc -l
    fi
}

# 处理PH任务
if [ ${#PH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理PH任务..."
    echo "===========================================" 
    
    for task_name in "${!PH_TASKS[@]}"; do
        checkpoint="${PH_TASKS[$task_name]}"
        process_task "$task_name" "$checkpoint" "PH"
    done
fi

# 处理MH任务
if [ ${#MH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理MH任务..."
    echo "===========================================" 
    
    for task_name in "${!MH_TASKS[@]}"; do
        checkpoint="${MH_TASKS[$task_name]}"
        process_task "$task_name" "$checkpoint" "MH"
    done
fi

# 处理低维任务
if [ ${#LOWDIM_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理低维任务..."
    echo "===========================================" 
    
    for task_name in "${!LOWDIM_TASKS[@]}"; do
        checkpoint="${LOWDIM_TASKS[$task_name]}"
        process_task "$task_name" "$checkpoint" "LOWDIM"
    done
fi

# 计算各类任务的平均FLOPS
ph_avg_original_flops=0
ph_avg_practical_flops=0
mh_avg_original_flops=0
mh_avg_practical_flops=0
lowdim_avg_original_flops=0
lowdim_avg_practical_flops=0

# 使用bc进行浮点数计算
if [ ${#PH_ORIGINAL_FLOPS[@]} -gt 0 ]; then
    ph_avg_original_flops=$(calculate_average "${PH_ORIGINAL_FLOPS[@]}")
    ph_avg_practical_flops=$(calculate_average "${PH_PRACTICAL_FLOPS[@]}")
fi

if [ ${#MH_ORIGINAL_FLOPS[@]} -gt 0 ]; then
    mh_avg_original_flops=$(calculate_average "${MH_ORIGINAL_FLOPS[@]}")
    mh_avg_practical_flops=$(calculate_average "${MH_PRACTICAL_FLOPS[@]}")
fi

if [ ${#LOWDIM_ORIGINAL_FLOPS[@]} -gt 0 ]; then
    lowdim_avg_original_flops=$(calculate_average "${LOWDIM_ORIGINAL_FLOPS[@]}")
    lowdim_avg_practical_flops=$(calculate_average "${LOWDIM_PRACTICAL_FLOPS[@]}")
fi

# 计算所有任务的平均FLOPS
all_original_flops=("${PH_ORIGINAL_FLOPS[@]}" "${MH_ORIGINAL_FLOPS[@]}" "${LOWDIM_ORIGINAL_FLOPS[@]}")
all_practical_flops=("${PH_PRACTICAL_FLOPS[@]}" "${MH_PRACTICAL_FLOPS[@]}" "${LOWDIM_PRACTICAL_FLOPS[@]}")

all_avg_original_flops=0
all_avg_practical_flops=0

if [ ${#all_original_flops[@]} -gt 0 ]; then
    all_avg_original_flops=$(calculate_average "${all_original_flops[@]}")
    all_avg_practical_flops=$(calculate_average "${all_practical_flops[@]}")
fi

# 安全计算百分比增加（避免除零）
safe_percentage() {
    local original=$1
    local practical=$2
    
    if (( $(echo "$original > 0" | bc -l) )); then
        echo "scale=2; ($practical - $original) / $original * 100" | bc -l
    else
        echo "0"
    fi
}

# 打印各类任务的平均FLOPS
echo "===========================================" 
echo "各类任务平均FLOPS统计"
echo "===========================================" 
echo "PH任务(${#PH_ORIGINAL_FLOPS[@]}个):"
echo "  平均原始FLOPS: $ph_avg_original_flops"
echo "  平均实际FLOPS: $ph_avg_practical_flops"
echo "  FLOPS增加比例: $(safe_percentage "$ph_avg_original_flops" "$ph_avg_practical_flops")%"
echo

echo "MH任务(${#MH_ORIGINAL_FLOPS[@]}个):"
echo "  平均原始FLOPS: $mh_avg_original_flops"
echo "  平均实际FLOPS: $mh_avg_practical_flops"
echo "  FLOPS增加比例: $(safe_percentage "$mh_avg_original_flops" "$mh_avg_practical_flops")%"
echo

echo "低维任务(${#LOWDIM_ORIGINAL_FLOPS[@]}个):"
echo "  平均原始FLOPS: $lowdim_avg_original_flops"
echo "  平均实际FLOPS: $lowdim_avg_practical_flops"
echo "  FLOPS增加比例: $(safe_percentage "$lowdim_avg_original_flops" "$lowdim_avg_practical_flops")%"
echo

echo "所有任务(${#all_original_flops[@]}个):"
echo "  平均原始FLOPS: $all_avg_original_flops"
echo "  平均实际FLOPS: $all_avg_practical_flops"
echo "  FLOPS增加比例: $(safe_percentage "$all_avg_original_flops" "$all_avg_practical_flops")%"
echo "===========================================" 

# 将平均FLOPS结果写入CSV文件
avg_flops_file="${RESULT_DIR}/average_flops_summary.csv"
echo "Task_Type,Count,Avg_Original_FLOPS,Avg_Practical_FLOPS,FLOPS_Increase_Percent" > "$avg_flops_file"
echo "PH,${#PH_ORIGINAL_FLOPS[@]},$ph_avg_original_flops,$ph_avg_practical_flops,$(safe_percentage "$ph_avg_original_flops" "$ph_avg_practical_flops")" >> "$avg_flops_file"
echo "MH,${#MH_ORIGINAL_FLOPS[@]},$mh_avg_original_flops,$mh_avg_practical_flops,$(safe_percentage "$mh_avg_original_flops" "$mh_avg_practical_flops")" >> "$avg_flops_file"
echo "LOWDIM,${#LOWDIM_ORIGINAL_FLOPS[@]},$lowdim_avg_original_flops,$lowdim_avg_practical_flops,$(safe_percentage "$lowdim_avg_original_flops" "$lowdim_avg_practical_flops")" >> "$avg_flops_file"
echo "ALL,${#all_original_flops[@]},$all_avg_original_flops,$all_avg_practical_flops,$(safe_percentage "$all_avg_original_flops" "$all_avg_practical_flops")" >> "$avg_flops_file"
echo "已保存平均FLOPS汇总到: $avg_flops_file"

echo "所有任务处理完成"
echo "===========================================" 

# 使用说明
usage() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --device device_name        指定要使用的设备 (默认: cuda:0)"
    echo "  --num_caches num            指定缓存数量 (默认: 10)"
    echo "  --metric metric             指定相似度指标 (默认: cosine)"
    echo "  --num_bu_blocks num         指定BU块数量 (默认: 5)"
    echo "  --task task_name            只处理指定的任务"
    echo "  --debug                     调试模式，只打印任务信息，不执行命令"
    echo
    echo "示例:"
    echo "  $0 --device cuda:1 --num_caches 10 --metric cosine --task block_pushing"
    echo "  $0 --num_bu_blocks 3 --task tool_hang_ph"
    echo "  $0 --task kitchen --debug"
}

# 如果没有参数则显示使用说明
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi
