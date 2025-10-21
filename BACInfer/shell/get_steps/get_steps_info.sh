#!/bin/bash

# 显示各个任务的BAC steps和BU算法后的steps信息
# 此脚本用于显示每个任务在cosine指标下，cache number为10时的steps选择情况

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"
echo "工作目录: $(pwd)"

# 默认参数
DEVICE="cuda:0"
NUM_CACHES="10"
METRIC="cosine"
NUM_BU_BLOCKS="5"
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

# 定义PH任务列表
# declare -A PH_TASKS=(
#     ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
#     ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
#     ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
#     ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
#     ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
#     ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
# )

# 定义MH任务列表
declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)

# 定义低维任务列表
# declare -A LOWDIM_TASKS=(
#     ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
#     ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
# )

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
echo "开始显示steps信息"
echo "===========================================" 
echo "设备: $DEVICE"
echo "缓存数量: $NUM_CACHES"
echo "相似度指标: $METRIC"
echo "BU块数量: $NUM_BU_BLOCKS"
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式: 开启 (只打印任务信息，不执行命令)"
fi
echo "===========================================" 

# 函数：处理一个任务的steps信息
process_task() {
    local task_name=$1
    local task_type=$2
    
    echo "===========================================" 
    echo "处理任务: $task_name (类型: $task_type)"
    echo "===========================================" 
    
    # 确定步骤目录
    local steps_dir="assets/${task_name}/original/optimal_steps/${METRIC}"
    if [ ! -d "$steps_dir" ]; then
        echo "错误: 步骤目录不存在: $steps_dir"
        return 1
    fi
    
    # 确定BU块文件
    local bu_blocks_file="assets/${task_name}/original/bu_block_selection/top_${NUM_BU_BLOCKS}_error_blocks.pkl"
    if [ ! -f "$bu_blocks_file" ]; then
        echo "错误: BU块文件不存在: $bu_blocks_file"
        return 1
    fi
    
    # 如果是调试模式，则只显示信息不执行
    if [ "$DEBUG_MODE" = true ]; then
        echo "调试模式: 跳过执行"
        return 0
    fi
    
    # 使用Python脚本处理steps信息
    python3 -c "
import os
import pickle
import re
from pathlib import Path

def get_layer_idx(block_key):
    match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
    if match:
        return int(match.group(1))
    match = re.match(r'decoder\.layers\.(\d+)\.dropout\d+', block_key)
    if match:
        return int(match.group(1))
    return -1

def get_block_type(block_key):
    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
    if match:
        return match.group(1)
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
    return block_key

# 读取原始steps
steps_dir = Path('$steps_dir')
original_steps = {}
for block_dir in steps_dir.iterdir():
    if block_dir.is_dir():
        block_name = block_dir.name
        steps_file = block_dir / f'optimal_steps_{block_name}_{$NUM_CACHES}_${METRIC}.pkl'
        if steps_file.exists():
            with open(steps_file, 'rb') as f:
                steps = pickle.load(f)
                original_steps[block_name] = steps

# 读取BU blocks
with open('$bu_blocks_file', 'rb') as f:
    bu_blocks = pickle.load(f)
    bu_blocks = [convert_to_block_format(block) for block in bu_blocks.keys()]

print('\nBU Blocks:')
print('-'*30)
for block in bu_blocks:
    print(f'- {block}')

# 计算practical steps
practical_steps = original_steps.copy()
ffn_blocks = [block for block in original_steps.keys() if 'ff_block' in block]

# 记录每个BU block的步数变化
step_changes = {}

for bu_block in bu_blocks:
    bu_layer = get_layer_idx(bu_block)
    deeper_ffn = [block for block in ffn_blocks if get_layer_idx(block) > bu_layer]
    
    all_deeper_steps = set()
    for deeper_block in deeper_ffn:
        if deeper_block in original_steps:
            all_deeper_steps.update(original_steps[deeper_block])
    
    if bu_block in practical_steps:
        current_steps = set(practical_steps[bu_block])
        new_steps = sorted(list(current_steps.union(all_deeper_steps)))
        practical_steps[bu_block] = new_steps
        
        # 记录步数变化
        added_steps = set(new_steps) - current_steps
        if added_steps:
            step_changes[bu_block] = {
                'original': sorted(list(current_steps)),
                'added': sorted(list(added_steps)),
                'final': new_steps
            }

# 打印结果
print('\nOriginal Steps (BAC):')
print('-'*30)
for block, steps in sorted(original_steps.items(), key=lambda x: (get_layer_idx(x[0]), get_block_type(x[0]))):
    print(f'{block}: {steps}')

print('\nPractical Steps (with BU=$NUM_BU_BLOCKS):')
print('-'*30)
for block, steps in sorted(practical_steps.items(), key=lambda x: (get_layer_idx(x[0]), get_block_type(x[0]))):
    if block in step_changes:
        print(f'{block}:')
        print(f'  Original: {step_changes[block][\"original\"]}')
        print(f'  Added:    {step_changes[block][\"added\"]}')
        print(f'  Final:    {step_changes[block][\"final\"]}')
    else:
        print(f'{block}: {steps}')
"
}

# 处理PH任务
if [ ${#PH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理PH任务..."
    echo "===========================================" 
    
    for task_name in "${!PH_TASKS[@]}"; do
        process_task "$task_name" "PH"
    done
fi

# 处理MH任务
if [ ${#MH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理MH任务..."
    echo "===========================================" 
    
    for task_name in "${!MH_TASKS[@]}"; do
        process_task "$task_name" "MH"
    done
fi

# 处理低维任务
if [ ${#LOWDIM_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理低维任务..."
    echo "===========================================" 
    
    for task_name in "${!LOWDIM_TASKS[@]}"; do
        process_task "$task_name" "LOWDIM"
    done
fi

echo "===========================================" 
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