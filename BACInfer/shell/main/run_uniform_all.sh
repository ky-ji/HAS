#!/bin/bash

# 统一步数模型评估脚本
# 用于在所有任务上运行统一步数模型的评估
# 支持PH、MH和lowdim三种类型的任务

# 默认配置
DEVICE="cuda:0"
TASK_TYPE="all"  # 可选: "ph", "mh", "lowdim", "all", "both" (ph+mh)
CACHE_MODE="threshold"
SKIP_VIDEO="--skip_video"
STEPS=7  # 默认步数
OUTPUT_DIR=""  # 默认输出目录，为空时会自动生成
SPECIFIC_TASKS=""  # 特定任务，为空时运行所有任务
SEEDS="0,1,2"  # 默认种子
FORCE_RECOMPUTE=false  # 是否强制重新计算，即使已有结果文件

# 定义任务列表
declare -a PH_TASKS=(
    "lift_ph"
    "can_ph"
    "square_ph"
    "pusht"
    "transport_ph"
    "tool_hang_ph"
)

declare -a MH_TASKS=(
    "lift_mh"
    "can_mh"
    "square_mh"
    "transport_mh"
)

declare -a LOWDIM_TASKS=(
    "block_pushing"
    "kitchen"
)

# 当前活跃的任务列表
declare -a ACTIVE_TASKS=()

# 定义任务和对应的检查点路径 (最大性能)
declare -A MAX_CHECKPOINTS_SEED0=(
    # PH任务
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0950-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2400-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2100-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.773.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.748.ckpt"
    
    # MH任务
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=3050-test_mean_score=1.000.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
    
    # LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED1=(
    # PH任务
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1150-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2200-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1800-test_mean_score=0.955.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt"
    
    # MH任务
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=1100-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0200-test_mean_score=0.773.ckpt"
    
    # LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/epoch=2700-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED2=(
    # PH任务
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3350-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt"
    
    # MH任务
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=3050-test_mean_score=0.955.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0350-test_mean_score=0.682.ckpt"
    
    # LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/epoch=1750-test_mean_score=0.574.ckpt"
)

# 定义任务和对应的检查点路径 (最后10个检查点平均)
declare -A AVG_CHECKPOINTS_SEED0=(
    # PH任务
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    
    # MH任务
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    
    # LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED1=(
    # PH任务
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    
    # MH任务
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    
    # LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED2=(
    # PH任务
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    
    # MH任务
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    
    # LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)

# 使用说明
usage() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --device device_name            指定要使用的设备 (默认: ${DEVICE})"
    echo "  --task_type type                指定要评估的任务类型 (ph, mh, lowdim, all, both) (默认: both)"
    echo "  --tasks task1,task2,...         指定特定的任务进行评估 (默认: 基于task_type的所有任务)"
    echo "  --steps N                       指定统一步数 (默认: 10)"
    echo "  --cache_mode mode               指定缓存模式 (threshold, original) (默认: threshold)"
    echo "  --seeds seed1,seed2,...         指定要评估的种子 (默认: 0,1,2)"
    echo "  --output_dir dir                指定输出目录 (默认: 自动生成)"
    echo "  --skip_video                    跳过视频渲染 (默认: 启用)"
    echo "  --with_video                    生成视频 (覆盖默认的skip_video)"
    echo "  --force                         强制重新计算，即使结果已存在"
    echo
    echo "示例:"
    echo "  $0 --device cuda:1 --task_type ph --tasks lift_ph,can_ph --steps 5 --seeds 0,1,2"
    echo "  $0 --device cuda:0 --task_type all --steps 10"
    echo "  $0 --device cuda:0 --task_type lowdim --with_video"
}

# 输出彩色标题
print_title() {
    echo -e "\033[1;34m$1\033[0m"
}

# 输出彩色小标题
print_subtitle() {
    echo -e "\033[1;36m$1\033[0m"
}

# 输出彩色信息
print_info() {
    echo -e "\033[0;32m$1\033[0m"
}

# 输出彩色警告
print_warning() {
    echo -e "\033[0;33m$1\033[0m"
}

# 输出彩色错误
print_error() {
    echo -e "\033[0;31m$1\033[0m"
}

# 输出彩色分隔线
print_separator() {
    echo -e "\033[0;35m----------------------------------------\033[0m"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --tasks)
            SPECIFIC_TASKS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --cache_mode)
            CACHE_MODE="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_video)
            SKIP_VIDEO="--skip_video"
            shift
            ;;
        --with_video)
            SKIP_VIDEO=""
            shift
            ;;
        --force)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# 设置活跃任务列表
if [ -n "$SPECIFIC_TASKS" ]; then
    # 如果指定了特定任务，使用它们
    IFS=',' read -ra ACTIVE_TASKS <<< "$SPECIFIC_TASKS"
else
    # 根据任务类型设置活跃任务
    case "$TASK_TYPE" in
        "ph")
            ACTIVE_TASKS=("${PH_TASKS[@]}")
            ;;
        "mh")
            ACTIVE_TASKS=("${MH_TASKS[@]}")
            ;;
        "lowdim")
            ACTIVE_TASKS=("${LOWDIM_TASKS[@]}")
            ;;
        "both")
            ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}")
            ;;
        "all")
            ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}" "${LOWDIM_TASKS[@]}")
            ;;
        *)
            echo "错误: 无效的任务类型 '$TASK_TYPE', 必须是 'ph', 'mh', 'lowdim', 'all' 或 'both'"
            exit 1
            ;;
    esac
fi

# 根据STEPS计算CACHE_THRESHOLD
CACHE_THRESHOLD=$((100 / STEPS))
if [ $CACHE_THRESHOLD -lt 1 ]; then
    CACHE_THRESHOLD=1
fi

# 解析种子列表
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# 设置输出目录
if [ -z "$OUTPUT_DIR" ]; then
    # 自动生成输出目录，使用结构化路径
    OUTPUT_DIR="results/benchmark/Uniform/${CACHE_MODE}_steps_${STEPS}"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 创建结果文件
RESULTS_FILE="${OUTPUT_DIR}/results.csv"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup,TaskType" > "$RESULTS_FILE"

# 打印实验配置信息
print_title "====================================================="
print_title "              统一步数模型评估                        "
print_title "====================================================="
print_info "设备: $DEVICE"
print_info "任务类型: $TASK_TYPE"
print_info "步数: $STEPS"
print_info "缓存模式: $CACHE_MODE"
if [ "$CACHE_MODE" = "threshold" ]; then
    print_info "缓存阈值: $CACHE_THRESHOLD (基于步数自动计算: 100/$STEPS)"
fi
print_info "活跃任务: ${ACTIVE_TASKS[*]}"
print_info "种子: ${SEED_ARRAY[*]}"
print_info "跳过视频: ${SKIP_VIDEO:+是}"
print_info "输出目录: $OUTPUT_DIR"
print_info "强制重新计算: $FORCE_RECOMPUTE"
print_title "====================================================="
echo ""

# 执行评估函数
execute_evaluation() {
    local task_name="$1"
    local seed="$2"
    local checkpoint="$3"
    local checkpoint_type="$4"
    
    # 确定任务类型
    local task_type=""
    if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
        task_type="ph"
    elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
        task_type="mh"
    elif [[ " ${LOWDIM_TASKS[*]} " =~ " ${task_name} " ]]; then
        task_type="lowdim"
    else
        print_warning "未知任务类型: $task_name"
        task_type="unknown"
    fi
    
    # 构建输出目录
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    local result_file="${task_output_dir}/fast_eval_log.json"
    local metrics_file="${task_output_dir}/eval_results.json"
    
    # 检查是否已有结果文件，且不是强制重新计算模式
    if [ -f "${result_file}" ] && [ -f "${metrics_file}" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_info "发现已存在的结果文件，直接读取: ${task_output_dir}"
        
        # 从现有文件中读取结果
        if [[ "$task_name" == "block_pushing" ]]; then
            # Block Pushing任务提取p1和p2指标
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info "从现有文件读取结果成功: p1=${bp_p1}, p2=${bp_p2}, 加速比=${speedup}"
            
        elif [[ "$task_name" == "kitchen" ]]; then
            # Kitchen任务提取p_1到p_4的成功率
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info "从现有文件读取结果成功: p1=${kitchen_p1}, p2=${kitchen_p2}, p3=${kitchen_p3}, p4=${kitchen_p4}, 加速比=${speedup}"
        else
            # 其他常规任务
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "Unifrom,${STEPS},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info "从现有文件读取结果成功: 成功率=${success_rate}, 加速比=${speedup}"
        fi
        
        print_separator
        return 0
    elif [ "$FORCE_RECOMPUTE" = true ]; then
        print_info "强制重新计算模式：将重新运行评估"
    fi
    
    # 创建输出目录
    mkdir -p "$task_output_dir"
    
    # 打印评估信息
    print_info "评估: ${task_name} (${task_type})"
    echo "缓存模式: ${CACHE_MODE}, 步数: ${STEPS}"
    if [ "$CACHE_MODE" = "threshold" ]; then
        echo "缓存阈值: ${CACHE_THRESHOLD} (基于步数自动计算: 100/$STEPS)"
    fi
    echo "种子: ${seed}"
    echo "检查点: ${checkpoint}"
    echo "检查点类型: ${checkpoint_type}"
    echo "输出目录: ${task_output_dir}"
    
    # 准备最优步骤目录(仅用于optimal模式)
    local optimal_steps_dir=""
    if [ "$CACHE_MODE" = "optimal" ]; then
        optimal_steps_dir="assets/${task_name}/original/optimal_steps/${METRIC}"
        
        # 检查最优步骤目录是否存在，不存在则计算
        if [ ! -d "$optimal_steps_dir" ] || [ "$FORCE_RECOMPUTE" = true ]; then
            echo "最优步骤目录不存在或强制重新计算，正在计算..."
            FORCE_FLAG=""
            if [ "$FORCE_RECOMPUTE" = true ]; then
                FORCE_FLAG="--force_recompute"
            fi
            python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py \
                -c "$checkpoint" \
                -o "assets/${task_name}/original" \
                -d "$DEVICE" \
                --num_caches "$STEPS" \
                --metrics "cosine" \
                $FORCE_FLAG
        fi
    fi
    
    # 构建评估命令的参数
    local eval_args=(
        --checkpoint "${checkpoint}"
        --output_dir "${task_output_dir}"
        --device "${DEVICE}"
        --cache_mode "${CACHE_MODE}"
    )
    
    # 根据缓存模式添加不同参数
    if [ "$CACHE_MODE" = "threshold" ]; then
        eval_args+=(--cache_threshold "${CACHE_THRESHOLD}")
    elif [ "$CACHE_MODE" = "optimal" ]; then
        eval_args+=(
            --optimal_steps_dir "${optimal_steps_dir}"
            --metric "cosine"
            --num_caches "${STEPS}"
            --num_bu_blocks 5
        )
    fi
    
    # 添加跳过视频参数
    if [ -n "$SKIP_VIDEO" ]; then
        eval_args+=($SKIP_VIDEO)
    fi
    
    # 执行评估
    python scripts/eval_fast_diffusion_policy.py "${eval_args[@]}"
    
    # 检查结果文件是否生成
    if [ -f "${result_file}" ]; then
        if [[ "$task_name" == "block_pushing" ]]; then
            # Block Pushing任务提取p1和p2指标
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info "评估完成: ${task_name} p1=${bp_p1}, p2=${bp_p2}, 加速比=${speedup}"
            
        elif [[ "$task_name" == "kitchen" ]]; then
            # Kitchen任务提取p_1到p_4的成功率
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info "评估完成: ${task_name} p1=${kitchen_p1}, p2=${kitchen_p2}, p3=${kitchen_p3}, p4=${kitchen_p4}, 加速比=${speedup}"
        else
            # 其他常规任务
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "Unifrom,${STEPS},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info "评估完成: 成功率=${success_rate}, 加速比=${speedup}"
        fi
    else
        print_warning "警告: 结果文件不存在: ${result_file}"
    fi
    
    print_separator
}

# 运行评估
for task_name in "${ACTIVE_TASKS[@]}"; do
    print_subtitle "处理任务: ${task_name}"
    
    for seed in "${SEED_ARRAY[@]}"; do
        for checkpoint_type in "max" "avg"; do
            # 获取对应种子和检查点类型的检查点路径
            if [ "$checkpoint_type" == "max" ]; then
                if [ "$seed" = "0" ]; then
                    checkpoint="${MAX_CHECKPOINTS_SEED0[$task_name]}"
                elif [ "$seed" = "1" ]; then
                    checkpoint="${MAX_CHECKPOINTS_SEED1[$task_name]}"
                elif [ "$seed" = "2" ]; then
                    checkpoint="${MAX_CHECKPOINTS_SEED2[$task_name]}"
                else
                    print_warning "未知种子: $seed, 跳过评估"
                    continue
                fi
            else
                if [ "$seed" = "0" ]; then
                    checkpoint="${AVG_CHECKPOINTS_SEED0[$task_name]}"
                elif [ "$seed" = "1" ]; then
                    checkpoint="${AVG_CHECKPOINTS_SEED1[$task_name]}"
                elif [ "$seed" = "2" ]; then
                    checkpoint="${AVG_CHECKPOINTS_SEED2[$task_name]}"
                else
                    print_warning "未知种子: $seed, 跳过评估"
                    continue
                fi
            fi
            
            # 检查检查点文件是否存在
            if [ ! -f "$checkpoint" ]; then
                print_warning "警告: 检查点文件不存在: $checkpoint, 跳过评估"
                continue
            fi
            
            # 执行评估
            execute_evaluation "$task_name" "$seed" "$checkpoint" "$checkpoint_type"
        done
    done
done

# 汇总结果
print_title "汇总结果..."
python - <<EOF
import pandas as pd
import numpy as np
import os

# 读取结果文件
results = pd.read_csv("$RESULTS_FILE")

# 获取任务类型
task_types = results['TaskType'].unique()

# 处理每种任务类型
for task_type in task_types:
    type_results = results[results['TaskType'] == task_type]
    
    print(f"\n\033[1;34m===== {task_type.upper()} 类型任务结果 =====\033[0m")
    
    # 获取所有任务名
    tasks = sorted(type_results['Task'].unique())
    
    # 计算每个任务的平均成功率和加速比
    print("\n\033[1;36m按任务和检查点类型的平均表现：\033[0m")
    for task in tasks:
        task_data = type_results[type_results['Task'] == task]
        print(f"\n任务: {task}")
        
        # 按检查点类型分组统计
        for ckpt_type in ["max", "avg"]:
            ckpt_data = task_data[task_data["CheckpointType"] == ckpt_type]
            if ckpt_data.empty:
                continue
                
            success_mean = ckpt_data['SuccessRate'].mean()
            success_std = ckpt_data['SuccessRate'].std()
            
            speedup_values = ckpt_data['Speedup'].replace("-", np.nan).astype(float)
            speedup_mean = speedup_values.mean()
            speedup_std = speedup_values.std()
            
            flops_values = ckpt_data['FLOPs'].replace("-", np.nan).astype(float)
            flops_mean = flops_values.mean()
            flops_std = flops_values.std()
            
            print(f"  检查点类型 {ckpt_type}:")
            print(f"    成功率: {success_mean:.3f} ± {success_std:.3f}")
            
            if not np.isnan(speedup_mean):
                print(f"    加速比: {speedup_mean:.2f}x ± {speedup_std:.2f}")
            else:
                print(f"    加速比: -")
                
            if not np.isnan(flops_mean):
                print(f"    FLOPs: {flops_mean:.2f} ± {flops_std:.2f}")
            else:
                print(f"    FLOPs: -")
    
    # 按种子汇总表现
    print("\n\033[1;36m按种子的表现汇总：\033[0m")
    for seed in sorted(type_results['Seed'].unique()):
        seed_data = type_results[type_results['Seed'] == seed]
        
        # 按检查点类型分组
        for ckpt_type in ["max", "avg"]:
            ckpt_seed_data = seed_data[seed_data["CheckpointType"] == ckpt_type]
            if ckpt_seed_data.empty:
                continue
                
            success_mean = ckpt_seed_data['SuccessRate'].mean()
            speedup_mean = ckpt_seed_data['Speedup'].replace("-", np.nan).astype(float).mean()
            
            print(f"  种子 {seed}, 检查点类型 {ckpt_type}:")
            print(f"    平均成功率: {success_mean:.3f}")
            if not np.isnan(speedup_mean):
                print(f"    平均加速比: {speedup_mean:.2f}x")
            else:
                print(f"    平均加速比: -")
    
    # 计算此类型任务的整体统计信息
    overall_success = type_results['SuccessRate'].mean()
    overall_speedup = type_results['Speedup'].replace("-", np.nan).astype(float).mean()
    
    print(f"\n{task_type.upper()} 类型任务整体统计:")
    print(f"  平均成功率: {overall_success:.3f}")
    
    if not np.isnan(overall_speedup):
        print(f"  平均加速比: {overall_speedup:.2f}x")
    else:
        print(f"  平均加速比: -")

# 计算所有任务的整体统计信息
overall_success = results['SuccessRate'].mean()
overall_speedup = results['Speedup'].replace("-", np.nan).astype(float).mean()

print(f"\n\033[1;34m===== 所有任务整体统计 =====\033[0m")
print(f"总平均成功率: {overall_success:.3f}")

if not np.isnan(overall_speedup):
    print(f"总平均加速比: {overall_speedup:.2f}x")
else:
    print(f"总平均加速比: -")

# 创建LaTeX表格格式的结果
print(f"\n\033[1;34m===== LaTeX表格输出 =====\033[0m")
for task_type in task_types:
    type_results = results[results['TaskType'] == task_type]
    
    print(f"\n{task_type.upper()} 任务表格:")
    
    # 按任务分组计算统计量
    for task in sorted(type_results['Task'].unique()):
        task_data = type_results[type_results['Task'] == task]
        
        # 计算平均成功率
        max_success = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
        avg_success = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
        
        # 计算平均加速比
        speedup_values = task_data['Speedup'].replace("-", np.nan).astype(float)
        avg_speedup = speedup_values.mean()
        
        if np.isnan(avg_speedup):
            print(f"{task} & {max_success:.3f}/{avg_success:.3f} & - \\\\")
        else:
            print(f"{task} & {max_success:.3f}/{avg_success:.3f} & {avg_speedup:.2f}x \\\\")

# 生成表格使用的特定格式输出
print(f"\n\033[1;34m===== 表格格式汇总数据 =====\033[0m")

# 获取所有任务和加速比数据
# 创建任务名映射
task_mappings = {
    # PH任务
    "lift_ph": "Lift$_{ph}$",
    "can_ph": "Can$_{ph}$",
    "square_ph": "Square$_{ph}$",
    "transport_ph": "Trans$_{ph}$",
    "tool_hang_ph": "Tool$_{ph}$",
    "pusht": "Push--T",
    
    # MH任务
    "lift_mh": "Lift$_{mh}$",
    "can_mh": "Can$_{mh}$",
    "square_mh": "Square$_{mh}$",
    "transport_mh": "Trans$_{mh}$",
    
    # LOWDIM任务
    "block_pushing_p1": "BP$_{p1}$",
    "block_pushing_p2": "BP$_{p2}$",
    "kitchen_p1": "Kit$_{p1}$",
    "kitchen_p2": "Kit$_{p2}$",
    "kitchen_p3": "Kit$_{p3}$",
    "kitchen_p4": "Kit$_{p4}$"
}

# 获取平均加速比
speedup_mean = results['Speedup'].replace("-", np.nan).astype(float).mean()
if np.isnan(speedup_mean):
    speedup_str = "--"
else:
    speedup_str = f"{speedup_mean:.2f}"

# 按任务类型生成表格所需数据
for steps in results['Steps'].unique():
    step_results = results[results['Steps'] == steps]
    
    # 处理PH任务
    ph_results = step_results[step_results['TaskType'] == 'ph']
    if not ph_results.empty:
        ph_data = {}
        for task in ['lift_ph', 'can_ph', 'square_ph', 'transport_ph', 'tool_hang_ph', 'pusht']:
            task_data = ph_results[ph_results['Task'] == task]
            if not task_data.empty:
                max_rate = task_data[task_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = task_data[task_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                ph_data[task] = f"{max_rate:.2f}/{avg_rate:.2f}"
            else:
                ph_data[task] = "--/--"
        
        # 生成PH任务的表格行
        ph_row = f"Uniform & {steps} & {ph_data.get('lift_ph', '--/--')} & {ph_data.get('can_ph', '--/--')} & {ph_data.get('square_ph', '--/--')} & {ph_data.get('transport_ph', '--/--')} & {ph_data.get('tool_hang_ph', '--/--')} & {ph_data.get('pusht', '--/--')} & -- & {speedup_str}"
        print("\n\033[1;36mPH任务表格行 (Uniform):\033[0m")
        print(ph_row)
    
    # 处理MH任务
    mh_results = step_results[step_results['TaskType'] == 'mh']
    if not mh_results.empty:
        mh_data = {}
        for task in ['lift_mh', 'can_mh', 'square_mh', 'transport_mh']:
            task_data = mh_results[mh_results['Task'] == task]
            if not task_data.empty:
                max_rate = task_data[task_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = task_data[task_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                mh_data[task] = f"{max_rate:.2f}/{avg_rate:.2f}"
            else:
                mh_data[task] = "--/--"
        
        # 生成MH任务的表格行
        mh_speedup = mh_results['Speedup'].replace("-", np.nan).astype(float).mean()
        if np.isnan(mh_speedup):
            mh_speedup_str = "--"
        else:
            mh_speedup_str = f"{mh_speedup:.2f}"
            
        mh_row = f"Uniform & {steps} & {mh_data.get('lift_mh', '--/--')} & {mh_data.get('can_mh', '--/--')} & {mh_data.get('square_mh', '--/--')} & {mh_data.get('transport_mh', '--/--')} & -- & {mh_speedup_str}"
        print("\n\033[1;36mMH任务表格行 (Uniform):\033[0m")
        print(mh_row)
    
    # 处理LOWDIM任务
    lowdim_results = step_results[step_results['TaskType'] == 'lowdim']
    if not lowdim_results.empty:
        lowdim_data = {}
        
        # 处理block_pushing的p1和p2
        bp_data = lowdim_results[lowdim_results['Task'].str.contains('block_pushing')]
        if not bp_data.empty:
            bp_p1_data = bp_data[bp_data['Task'] == 'block_pushing_p1']
            if not bp_p1_data.empty:
                max_rate = bp_p1_data[bp_p1_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = bp_p1_data[bp_p1_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                lowdim_data['block_pushing_p1'] = f"{max_rate:.2f}/{avg_rate:.2f}"
            
            bp_p2_data = bp_data[bp_data['Task'] == 'block_pushing_p2']
            if not bp_p2_data.empty:
                max_rate = bp_p2_data[bp_p2_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = bp_p2_data[bp_p2_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                lowdim_data['block_pushing_p2'] = f"{max_rate:.2f}/{avg_rate:.2f}"
        
        # 处理kitchen的p1到p4
        kitchen_data = lowdim_results[lowdim_results['Task'].str.contains('kitchen')]
        for i in range(1, 5):
            kitchen_pi_data = kitchen_data[kitchen_data['Task'] == f'kitchen_p{i}']
            if not kitchen_pi_data.empty:
                max_rate = kitchen_pi_data[kitchen_pi_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = kitchen_pi_data[kitchen_pi_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                lowdim_data[f'kitchen_p{i}'] = f"{max_rate:.2f}/{avg_rate:.2f}"
            else:
                lowdim_data[f'kitchen_p{i}'] = "--/--"
        
        # 生成LOWDIM任务的表格行
        lowdim_speedup = lowdim_results['Speedup'].replace("-", np.nan).astype(float).mean()
        if np.isnan(lowdim_speedup):
            lowdim_speedup_str = "--"
        else:
            lowdim_speedup_str = f"{lowdim_speedup:.2f}"
            
        lowdim_row = f"Uniform & {steps} & {lowdim_data.get('block_pushing_p1', '--/--')} & {lowdim_data.get('block_pushing_p2', '--/--')} & {lowdim_data.get('kitchen_p1', '--/--')} & {lowdim_data.get('kitchen_p2', '--/--')} & {lowdim_data.get('kitchen_p3', '--/--')} & {lowdim_data.get('kitchen_p4', '--/--')} & -- & {lowdim_speedup_str}"
        print("\n\033[1;36mLOWDIM任务表格行 (Uniform):\033[0m")
        print(lowdim_row)

# 保存汇总文件
summary_file = "$RESULTS_FILE".replace(".csv", "_summary.csv")
summary_data = []

# 按任务和检查点类型汇总
for task in sorted(results['Task'].unique()):
    task_data = results[results['Task'] == task]
    task_type = task_data['TaskType'].iloc[0]
    
    for ckpt_type in ["max", "avg"]:
        ckpt_data = task_data[task_data["CheckpointType"] == ckpt_type]
        if ckpt_data.empty:
            continue
            
        success_mean = ckpt_data['SuccessRate'].mean()
        speedup_values = ckpt_data['Speedup'].replace("-", np.nan).astype(float)
        speedup_mean = speedup_values.mean()
        flops_values = ckpt_data['FLOPs'].replace("-", np.nan).astype(float)
        flops_mean = flops_values.mean()
        
        summary_data.append({
            "Task": task,
            "TaskType": task_type,
            "CheckpointType": ckpt_type,
            "Steps": $STEPS,
            "CacheMode": "${CACHE_MODE}",
            "SuccessRate": success_mean,
            "Speedup": speedup_mean if not np.isnan(speedup_mean) else "-",
            "FLOPs": flops_mean if not np.isnan(flops_mean) else "-"
        })

# 创建和保存汇总文件
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_file, index=False)
print(f"\n汇总数据已保存至: {summary_file}")
EOF

print_title "统一步数评估完成！结果已保存到以下文件:"
echo "- 主结果文件: $RESULTS_FILE"
echo "- 汇总结果文件: ${RESULTS_FILE//.csv/_summary.csv}"
echo "- 详细结果: $OUTPUT_DIR/{task_name}/seed{seed}_{checkpoint_type}/"