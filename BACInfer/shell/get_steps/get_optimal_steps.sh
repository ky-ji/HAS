#!/bin/bash

# 获取所有任务的optimal steps脚本
# 此脚本将为所有任务(PH/MH/低维任务)计算最优缓存步骤

# 默认参数
DEVICE="cuda:0"
NUM_CACHES="10"
METRICS="cosine"
FORCE_RECOMPUTE="--force_recompute"
DEBUG_MODE=false  # 调试模式参数

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
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --no_force)
            FORCE_RECOMPUTE=""
            shift
            ;;
        --ph_only)
            PH_ONLY=true
            shift
            ;;
        --mh_only)
            MH_ONLY=true
            shift
            ;;
        --lowdim_only)
            LOWDIM_ONLY=true
            shift
            ;;
        --task)
            SINGLE_TASK="$2"
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

# # 定义PH任务列表及其检查点
declare -A PH_TASKS=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3350-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt"
)

# 定义MH任务列表及其检查点
declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=3050-test_mean_score=0.955.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)

# 定义低维任务列表及其检查点
declare -A LOWDIM_TASKS=(
    # Block Pushing (BP)任务检查点
    ["block_pushing"]="checkpoint/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    
    # Kitchen任务检查点
    ["kitchen"]="checkpoint/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)



# 定义PH任务列表及其检查点
declare -A PH_TASKS=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)

# 定义MH任务列表及其检查点
declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)

# 定义低维任务列表及其检查点
declare -A LOWDIM_TASKS=(
    # Block Pushing (BP)任务检查点
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    
    # Kitchen任务检查点
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)






# 处理单个任务的情况
if [ ! -z "$SINGLE_TASK" ]; then
    if [[ -v PH_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理PH任务: $SINGLE_TASK"
        # 创建一个新的关联数组，只包含指定任务
        declare -A PH_TASKS_FILTERED
        PH_TASKS_FILTERED["$SINGLE_TASK"]="${PH_TASKS[$SINGLE_TASK]}"
        PH_TASKS=()
        # 正确方式是复制每一个键值对
        for key in "${!PH_TASKS_FILTERED[@]}"; do
            PH_TASKS["$key"]="${PH_TASKS_FILTERED[$key]}"
        done
        # 清空其他任务数组
        MH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v MH_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理MH任务: $SINGLE_TASK"
        # 创建一个新的关联数组，只包含指定任务
        declare -A MH_TASKS_FILTERED
        MH_TASKS_FILTERED["$SINGLE_TASK"]="${MH_TASKS[$SINGLE_TASK]}"
        MH_TASKS=()
        # 正确方式是复制每一个键值对
        for key in "${!MH_TASKS_FILTERED[@]}"; do
            MH_TASKS["$key"]="${MH_TASKS_FILTERED[$key]}"
        done
        # 清空其他任务数组
        PH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v LOWDIM_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理低维任务: $SINGLE_TASK"
        # 创建一个新的关联数组，只包含指定任务
        declare -A LOWDIM_TASKS_FILTERED
        LOWDIM_TASKS_FILTERED["$SINGLE_TASK"]="${LOWDIM_TASKS[$SINGLE_TASK]}"
        LOWDIM_TASKS=()
        # 正确方式是复制每一个键值对
        for key in "${!LOWDIM_TASKS_FILTERED[@]}"; do
            LOWDIM_TASKS["$key"]="${LOWDIM_TASKS_FILTERED[$key]}"
        done
        # 清空其他任务数组
        PH_TASKS=()
        MH_TASKS=()
    else
        echo "错误: 找不到任务 $SINGLE_TASK"
        exit 1
    fi
elif [ "$PH_ONLY" = true ]; then
    echo "仅处理PH任务"
    declare -A MH_TASKS=()
    declare -A LOWDIM_TASKS=()
elif [ "$MH_ONLY" = true ]; then
    echo "仅处理MH任务"
    declare -A PH_TASKS=()
    declare -A LOWDIM_TASKS=()
elif [ "$LOWDIM_ONLY" = true ]; then
    echo "仅处理低维任务"
    declare -A PH_TASKS=()
    declare -A MH_TASKS=()
fi

# 显示任务信息
echo "===========================================" 
echo "开始获取最优缓存步骤"
echo "===========================================" 
echo "设备: $DEVICE"
echo "缓存数量: $NUM_CACHES"
echo "相似度指标: $METRICS"
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式: 开启 (只打印任务信息，不执行命令)"
fi
if [ ! -z "$FORCE_RECOMPUTE" ]; then
    echo "强制重新计算: 是"
else
    echo "强制重新计算: 否"
fi
echo "===========================================" 

# 创建assets目录
mkdir -p assets

# 处理PH任务
if [ ${#PH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理PH任务..."
    echo "===========================================" 
    
    for task_name in "${!PH_TASKS[@]}"; do
        checkpoint="${PH_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo "警告: 检查点文件不存在: $checkpoint, 跳过任务"
            continue
        fi

        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo "处理PH任务: $task_name"
        echo "检查点: $checkpoint"
        echo "输出目录: $output_dir"
        
        # 调试模式下跳过执行
        if [ "$DEBUG_MODE" = true ]; then
            echo "调试模式: 跳过执行"
            continue
        fi
        
        # 运行命令
        cmd="python -m BACInfer.analysis.get_optimal_cache_update_steps -c $checkpoint -o $output_dir -d $DEVICE --num_caches $NUM_CACHES --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo "执行命令: $cmd"
        eval $cmd
        
        echo "任务 ${task_name} 完成"
        echo "-------------------------------------------"
    done
fi

# 处理MH任务
if [ ${#MH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理MH任务..."
    echo "===========================================" 
    
    for task_name in "${!MH_TASKS[@]}"; do
        checkpoint="${MH_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo "警告: 检查点文件不存在: $checkpoint, 跳过任务"
            continue
        fi
        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo "处理MH任务: $task_name"
        echo "检查点: $checkpoint"
        echo "输出目录: $output_dir"
        
        # 调试模式下跳过执行
        if [ "$DEBUG_MODE" = true ]; then
            echo "调试模式: 跳过执行"
            continue
        fi
        
        # 运行命令
        cmd="python -m BACInfer.analysis.get_optimal_cache_update_steps -c $checkpoint -o $output_dir -d $DEVICE --num_caches $NUM_CACHES --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo "执行命令: $cmd"
        eval $cmd
        
        echo "任务 ${task_name} 完成"
        echo "-------------------------------------------"
    done
fi

# 处理低维任务
if [ ${#LOWDIM_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理低维任务..."
    echo "===========================================" 
    
    for task_name in "${!LOWDIM_TASKS[@]}"; do
        checkpoint="${LOWDIM_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo "警告: 检查点文件不存在: $checkpoint, 跳过任务"
            continue
        fi
        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo "处理低维任务: $task_name"
        echo "检查点: $checkpoint"
        echo "输出目录: $output_dir"
        
        # 调试模式下跳过执行
        if [ "$DEBUG_MODE" = true ]; then
            echo "调试模式: 跳过执行"
            continue
        fi
        
        # 运行命令
        cmd="python -m BACInfer.analysis.get_optimal_cache_update_steps -c $checkpoint -o $output_dir -d $DEVICE --num_caches $NUM_CACHES --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo "执行命令: $cmd"
        eval $cmd
        
        echo "任务 ${task_name} 完成"
        echo "-------------------------------------------"
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
    echo "  --num_caches list           指定要计算的缓存数量，逗号分隔 (默认: 5,8,10,20)"
    echo "  --metrics list              指定要计算的相似度指标，逗号分隔 (默认: cosine,l1,mse)"
    echo "  --no_force                  不强制重新计算"
    echo "  --ph_only                   只处理PH任务"
    echo "  --mh_only                   只处理MH任务"
    echo "  --lowdim_only               只处理低维任务(block_pushing和kitchen)"
    echo "  --task task_name            只处理指定的任务"
    echo "  --debug                     调试模式，只打印任务信息，不执行命令"
    echo
    echo "示例:"
    echo "  $0 --device cuda:1 --num_caches 8,10,15 --metrics cosine --task block_pushing"
    echo "  $0 --device cuda:0 --lowdim_only --num_caches 8,10,15"
    echo "  $0 --lowdim_only --task kitchen --debug"
}

# 如果没有参数则显示使用说明
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi
