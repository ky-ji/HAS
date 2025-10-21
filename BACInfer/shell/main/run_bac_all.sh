#!/bin/bash

# BAC模型评估脚本
# 用于在所有任务上运行BAC模型的评估
# 支持PH、MH和lowdim三种类型的任务
# 使用optimal模式和指定的num_bu_blocks值
# 
# bu参数说明:
# num_bu_blocks - block update的块数，值为0表示对每个块独立使用自适应步骤
#               - 值为大于0的整数表示使用block update机制，将模型分为指定数量的块
#               - 默认值为5
#
# 输出目录格式:
# results/${TASK_NAME}/${CACHE_MODE}/${METRIC}_caches${STEPS}_bu${NUM_BU_BLOCKS}_seed${SEED}_${CHECKPOINT_TYPE}

# 默认配置
DEVICE="cuda:3"
TASK_TYPE="all"  # 可选: "ph", "mh", "lowdim", "all", "both" (ph+mh)
CACHE_MODE="optimal"
SKIP_VIDEO="--skip_video"
STEPS=10  # 默认步数
NUM_BU_BLOCKS=5  # BAC模型的块数
METRIC="cosine"  # 默认度量标准
OUTPUT_DIR=""  # 默认输出目录，为空时会自动生成
SPECIFIC_TASKS=""  # 特定任务，为空时运行所有任务
SEEDS="0,1,2"  # 默认种子
FORCE_RECOMPUTE=false  # 是否强制重新计算，即使已有结果文件
CHECKPOINT_TYPE="all"  # 使用最大性能检查点，可选值: "max", "avg", "all"

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
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
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
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
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
    echo "  --device device_name         指定要使用的设备 (默认: ${DEVICE})"
    echo "  --steps N                    指定缓存步数 (默认: 10)"
    echo "  --seeds seed1,seed2,...      指定要评估的种子 (默认: 0,1,2)"
    echo "  --metric metric_name         指定度量标准 (默认: cosine)"
    echo "  --cache_mode mode            指定缓存模式 (optimal, threshold) (默认: optimal)"
    echo "  --skip_video                 跳过视频渲染 (默认: 启用)"
    echo "  --output_dir dir             指定输出目录 (默认: 自动生成)"
    echo "  --force                      强制重新计算，即使结果已存在"
    echo "  --checkpoint_type type       指定检查点类型 (avg, max, all) (默认: avg)"
    echo "                               'all'选项将同时运行avg和max两种类型"
    echo "  --task task1,task2,...       指定要评估的特定任务"
    echo "  --ph_only                    仅评估PH类型任务"
    echo "  --mh_only                    仅评估MH类型任务"
    echo "  --lowdim_only                仅评估低维任务 (block_pushing和kitchen)"
    echo "  --bp_only                    仅评估block_pushing任务"
    echo "  --kitchen_only               仅评估kitchen任务"
    echo
    echo "示例:"
    echo "  $0 --device cuda:0 --steps 5 --seeds 0,1,2 --metric l1"
    echo "  $0 --device cuda:0 --ph_only --checkpoint_type max"
    echo "  $0 --device cuda:0 --task lift_ph,can_ph --checkpoint_type all"
}

# 解析命令行参数
while [ "$#" -gt 0 ]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --cache_mode)
            CACHE_MODE="$2"
            shift 2
            ;;
        --num_bu_blocks)
            NUM_BU_BLOCKS="$2"
            shift 2
            ;;
        --skip_video)
            SKIP_VIDEO="--skip_video"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task)
            SPECIFIC_TASKS="$2"
            shift 2
            ;;
        --force)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --checkpoint_type)
            CHECKPOINT_TYPE="$2"
            # 检查是否是有效的检查点类型
            if [ "$CHECKPOINT_TYPE" != "max" ] && [ "$CHECKPOINT_TYPE" != "avg" ] && [ "$CHECKPOINT_TYPE" != "all" ]; then
                print_error "无效的检查点类型: $CHECKPOINT_TYPE"
                print_error "有效选项: max, avg, all"
                exit 1
            fi
            shift 2
            ;;
        --ph_only)
            TASK_TYPE="ph"
            shift
            ;;
        --mh_only)
            TASK_TYPE="mh"
            shift
            ;;
        --lowdim_only)
            TASK_TYPE="lowdim"
            shift
            ;;
        --bp_only)
            TASK_TYPE="block_pushing"
            shift
            ;;
        --kitchen_only)
            TASK_TYPE="kitchen"
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

# 设置输出目录
if [ -z "$OUTPUT_DIR" ]; then
    # 汇总结果的主目录，包含完整路径信息
    OUTPUT_DIR="results/benchmark/BAC/${CACHE_MODE}_${METRIC}_caches${STEPS}_bu${NUM_BU_BLOCKS}"
fi

# 创建汇总结果目录
mkdir -p "$OUTPUT_DIR"

# 定义RESULTS_DIR变量存储父目录路径（用于后续结果汇总）
RESULTS_DIR="$OUTPUT_DIR"
RESULTS_FILE="${RESULTS_DIR}/results.csv"

# 创建结果文件
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$RESULTS_FILE"

# 为每种任务类型创建单独的结果文件
PH_RESULTS_FILE="${RESULTS_DIR}/ph_results.csv"
MH_RESULTS_FILE="${RESULTS_DIR}/mh_results.csv"
BP_RESULTS_FILE="${RESULTS_DIR}/bp_results.csv"
KITCHEN_RESULTS_FILE="${RESULTS_DIR}/kitchen_results.csv"

# 初始化各类型任务的结果文件
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$PH_RESULTS_FILE"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$MH_RESULTS_FILE"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$BP_RESULTS_FILE"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$KITCHEN_RESULTS_FILE"

# 根据任务类型设置活跃任务列表
if [ "$TASK_TYPE" = "all" ]; then
    ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}" "${LOWDIM_TASKS[@]}")
elif [ "$TASK_TYPE" = "both" ]; then
    ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}")
elif [ "$TASK_TYPE" = "ph" ]; then
    ACTIVE_TASKS=("${PH_TASKS[@]}")
elif [ "$TASK_TYPE" = "mh" ]; then
    ACTIVE_TASKS=("${MH_TASKS[@]}")
elif [ "$TASK_TYPE" = "lowdim" ]; then
    ACTIVE_TASKS=("${LOWDIM_TASKS[@]}")
elif [ "$TASK_TYPE" = "block_pushing" ]; then
    ACTIVE_TASKS=("block_pushing")
elif [ "$TASK_TYPE" = "kitchen" ]; then
    ACTIVE_TASKS=("kitchen")
fi

# 如果指定了特定任务，则仅对这些任务进行评估
if [ ! -z "$SPECIFIC_TASKS" ]; then
    IFS=',' read -ra SPECIFIC_TASKS_ARRAY <<< "$SPECIFIC_TASKS"
    ACTIVE_TASKS=("${SPECIFIC_TASKS_ARRAY[@]}")
fi

# 输出配置信息
echo "====================================================="
echo "BAC模型评估配置"
echo "====================================================="
echo "设备: $DEVICE"
echo "步数: $STEPS"
echo "种子: $SEEDS"
echo "指标: $METRIC"
echo "跳过视频: ${SKIP_VIDEO:+是}"
echo "任务类型: $TASK_TYPE"
echo "缓存模式: $CACHE_MODE"
echo "块数: $NUM_BU_BLOCKS"
echo "输出目录: $OUTPUT_DIR"
echo "检查点类型: $CHECKPOINT_TYPE"
echo "强制重新计算: $FORCE_RECOMPUTE"
echo "活跃任务: ${ACTIVE_TASKS[*]}"
echo "====================================================="

# 定义颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

# 打印带颜色的标题
print_title() {
    echo -e "${RED}$1${NC}"
}

# 打印带颜色的子标题
print_subtitle() {
    echo -e "${BLUE}$1${NC}"
}

# 打印带颜色的信息
print_info() {
    echo -e "${GREEN}$1${NC}"
}

# 打印带颜色的警告
print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# 打印带颜色的错误
print_error() {
    echo -e "${RED}$1${NC}"
}

# 打印带颜色的分隔线
print_separator() {
    echo -e "${BLUE}----------------------------------------${NC}"
}

# 执行单个评估
execute_single_evaluation() {
    local task_name="$1"
    local seed="$2"
    local checkpoint_type="$3"
    
    # 构建输出目录
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    local result_file="${task_output_dir}/fast_eval_log.json"
    local metrics_file="${task_output_dir}/eval_results.json"
    
    # 检查是否已有结果文件，且不是强制重新计算模式
    if [ -f "${result_file}" ] && [ -f "${metrics_file}" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_info "发现已存在的结果文件，直接读取: ${task_output_dir}"
        
        # 从结果文件中提取评估结果
        # 特殊处理block_pushing和kitchen任务，它们有多个子任务结果
        if [ "$task_name" = "block_pushing" ]; then
            # 提取p1和p2的成功率
            local BP_P1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local BP_P2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 记录结果到任务类型特定的结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            
            print_info "评估完成: ${task_name} p1=${BP_P1}, p2=${BP_P2} (BAC模式, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type})"
        elif [ "$task_name" = "kitchen" ]; then
            # 提取p_1到p_4的成功率
            local KITCHEN_P1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 记录结果到任务类型特定的结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            
            print_info "评估完成: ${task_name} p1=${KITCHEN_P1}, p2=${KITCHEN_P2}, p3=${KITCHEN_P3}, p4=${KITCHEN_P4} (BAC模式, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type})"
        else
            # 普通任务的处理
            local SUCCESS_RATE=$(grep "mean_score" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 根据任务类型记录到对应结果文件
            if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$PH_RESULTS_FILE"
            elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$MH_RESULTS_FILE"
            fi
            
            print_info "评估完成: ${task_name} (BAC模式, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type}, 成功率: ${SUCCESS_RATE})"
        fi
        
        print_separator
        return 0
    fi
    
    # 选择检查点
    local checkpoint=""
    if [ "$checkpoint_type" = "max" ]; then
        if [ "$seed" = "0" ]; then
            checkpoint="${MAX_CHECKPOINTS_SEED0[$task_name]}"
        elif [ "$seed" = "1" ]; then
            checkpoint="${MAX_CHECKPOINTS_SEED1[$task_name]}"
        elif [ "$seed" = "2" ]; then
            checkpoint="${MAX_CHECKPOINTS_SEED2[$task_name]}"
        fi
    else # avg
        if [ "$seed" = "0" ]; then
            checkpoint="${AVG_CHECKPOINTS_SEED0[$task_name]}"
        elif [ "$seed" = "1" ]; then
            checkpoint="${AVG_CHECKPOINTS_SEED1[$task_name]}"
        elif [ "$seed" = "2" ]; then
            checkpoint="${AVG_CHECKPOINTS_SEED2[$task_name]}"
        fi
    fi
    
    if [ -z "$checkpoint" ]; then
        print_warning "未找到任务 $task_name 和种子 $seed 的检查点，跳过评估"
        return 1
    fi
    
    # 准备最优步骤目录
    local OPTIMAL_STEPS_DIR="assets/${task_name}/original/optimal_steps/${METRIC}"
    
    # 计算最优缓存步骤
    print_info "计算最优缓存步骤: 步数=${STEPS}, 任务=${task_name}, 种子=${seed}, 检查点类型=${checkpoint_type}"
    
    # 检查最优步骤目录是否存在
    if [ ! -d "$OPTIMAL_STEPS_DIR" ]; then
        print_info "最优步骤目录不存在，正在计算..."
        FORCE_FLAG=""
        if [ "$FORCE_RECOMPUTE" = true ]; then
            FORCE_FLAG="--force_recompute"
        fi
        python -m BACInfer.analysis.get_optimal_cache_update_steps \
            -c "$checkpoint" \
            -o "assets/${task_name}/original" \
            -d "$DEVICE" \
            --num_caches "$STEPS" \
            --metrics "$METRIC" \
            $FORCE_FLAG
    fi
    
    # 检查计算后最优步骤目录是否存在
    if [ ! -d "$OPTIMAL_STEPS_DIR" ]; then
        print_error "错误: 无法找到或创建最优步骤目录: $OPTIMAL_STEPS_DIR"
        print_error "任务 ${task_name} 的评估失败，退出..."
        return 1
    fi
    
    # 检查目录中是否存在相关步骤文件
    local step_files_count=$(find "$OPTIMAL_STEPS_DIR" -name "*_${STEPS}_${METRIC}.pkl" 2>/dev/null | wc -l)
    if [ "$step_files_count" -eq 0 ]; then
        print_error "错误: 未找到${STEPS}步的最优步骤文件: $OPTIMAL_STEPS_DIR/*_${STEPS}_${METRIC}.pkl"
        print_error "任务 ${task_name} 的评估失败，退出..."
        return 1
    fi
    
    # 设置BAC评估参数和输出目录
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    
    print_info "评估任务: ${task_name}, 模式: ${CACHE_MODE}, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type}"
    print_info "检查点: ${checkpoint}"
    print_info "输出目录: ${task_output_dir}"
    print_info "最优步骤目录: ${OPTIMAL_STEPS_DIR}"
    
    # 检查是否已存在结果数据，如果存在则跳过评估
    if [ -f "${task_output_dir}/fast_eval_log.json" ] && [ -f "${task_output_dir}/eval_results.json" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_warning "评估结果已存在，跳过重新计算: ${task_output_dir}"
    else
        # 执行评估
        local eval_args=(
            --checkpoint "${checkpoint}"
            --output_dir "${task_output_dir}"
            --device "${DEVICE}"
            --cache_mode "${CACHE_MODE}"
            --optimal_steps_dir "${OPTIMAL_STEPS_DIR}"
            --metric "${METRIC}"
            --num_caches "${STEPS}"
            --num_bu_blocks "${NUM_BU_BLOCKS}"
            ${SKIP_VIDEO}
        )
        python -m BACInfer.scripts.eval_fast_diffusion_policy "${eval_args[@]}"
    fi
    
    # 获取任务类型对应的结果文件
    local TASK_TYPE_FILE=""
    if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
        TASK_TYPE_FILE="$PH_RESULTS_FILE"
    elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
        TASK_TYPE_FILE="$MH_RESULTS_FILE"
    elif [ "$task_name" = "block_pushing" ]; then
        TASK_TYPE_FILE="$BP_RESULTS_FILE"
    elif [ "$task_name" = "kitchen" ]; then
        TASK_TYPE_FILE="$KITCHEN_RESULTS_FILE"
    fi
    
    # 确保任务类型结果文件存在
    if [ ! -f "$TASK_TYPE_FILE" ]; then
        echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$TASK_TYPE_FILE"
    fi
    
    # 从结果文件中提取评估结果
    if [ -f "${task_output_dir}/fast_eval_log.json" ]; then
        # 特殊处理block_pushing和kitchen任务，它们有多个子任务结果
        if [ "$task_name" = "block_pushing" ]; then
            # 提取p1和p2的成功率
            local BP_P1=$(grep "\"test/p1\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local BP_P2=$(grep "\"test/p2\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 记录结果到任务类型特定的结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            
            print_info "评估完成: ${task_name} p1=${BP_P1}, p2=${BP_P2} (BAC模式, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type})"
        elif [ "$task_name" = "kitchen" ]; then
            # 提取p_1到p_4的成功率
            local KITCHEN_P1=$(grep "\"test/p_1\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P2=$(grep "\"test/p_2\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P3=$(grep "\"test/p_3\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P4=$(grep "\"test/p_4\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 记录结果到任务类型特定的结果文件
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            
            print_info "评估完成: ${task_name} p1=${KITCHEN_P1}, p2=${KITCHEN_P2}, p3=${KITCHEN_P3}, p4=${KITCHEN_P4} (BAC模式, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type})"
        else
            # 普通任务的处理
            local SUCCESS_RATE=$(grep "mean_score" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 记录结果到主结果文件
            echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 根据任务类型记录到对应结果文件
            if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$PH_RESULTS_FILE"
            elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$MH_RESULTS_FILE"
            fi
            
            print_info "评估完成: ${task_name} (BAC模式, 步数: ${STEPS}, 种子: ${seed}, 检查点类型: ${checkpoint_type}, 成功率: ${SUCCESS_RATE})"
        fi
    else
        print_warning "评估失败，未找到结果文件: ${task_output_dir}/fast_eval_log.json"
    fi
}

# 执行评估
print_title "开始BAC模型评估"
print_info "输出目录: $OUTPUT_DIR"
print_info "结果文件: $RESULTS_FILE"

# 根据CHECKPOINT_TYPE决定要评估的检查点类型列表
declare -a CHECKPOINT_TYPES=()
if [ "$CHECKPOINT_TYPE" = "all" ]; then
    CHECKPOINT_TYPES=("max" "avg")
else
    CHECKPOINT_TYPES=("$CHECKPOINT_TYPE")
fi

# 遍历所有活跃任务和种子
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"
for task_name in "${ACTIVE_TASKS[@]}"; do
    print_subtitle "评估任务: $task_name"
    for seed in "${SEEDS_ARRAY[@]}"; do
        print_info "使用种子: $seed"
        # 遍历要评估的检查点类型
        for current_checkpoint_type in "${CHECKPOINT_TYPES[@]}"; do
            print_info "使用检查点类型: $current_checkpoint_type"
            execute_single_evaluation "$task_name" "$seed" "$current_checkpoint_type"
        done
    done
done

print_title "BAC模型评估完成"
print_info "最终结果文件:"
print_info "- 主结果文件: $RESULTS_FILE"

# 统一生成汇总表数据
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
echo "TaskType,Method,Steps,Task,MaxSuccessRate,AvgSuccessRate,FLOPs,Speedup" > "$SUMMARY_FILE"

# 调用Python处理所有任务数据并生成符合表格格式的汇总
python3 - <<EOF
import pandas as pd
import numpy as np
import os
import glob
import json

# 使用RESULTS_DIR变量获取结果目录
result_dir = "${RESULTS_DIR}"
print(f"处理结果目录: {result_dir}")

# 获取所有结果文件
all_results = []

# 尝试读取PH结果
ph_file = os.path.join(result_dir, "ph_results.csv")
if os.path.exists(ph_file):
    try:
        ph_df = pd.read_csv(ph_file)
        ph_df['TaskType'] = 'ph'
        all_results.append(ph_df)
        print(f"成功读取PH结果: {ph_file}")
    except Exception as e:
        print(f"读取PH结果文件出错: {e}")

# 尝试读取MH结果
mh_file = os.path.join(result_dir, "mh_results.csv")
if os.path.exists(mh_file):
    try:
        mh_df = pd.read_csv(mh_file)
        mh_df['TaskType'] = 'mh'
        all_results.append(mh_df)
        print(f"成功读取MH结果: {mh_file}")
    except Exception as e:
        print(f"读取MH结果文件出错: {e}")

# 尝试读取BP结果
bp_file = os.path.join(result_dir, "bp_results.csv")
if os.path.exists(bp_file):
    try:
        bp_df = pd.read_csv(bp_file)
        bp_df['TaskType'] = 'bp'
        all_results.append(bp_df)
        print(f"成功读取BP结果: {bp_file}")
    except Exception as e:
        print(f"读取BP结果文件出错: {e}")

# 尝试读取Kitchen结果
kitchen_file = os.path.join(result_dir, "kitchen_results.csv")
if os.path.exists(kitchen_file):
    try:
        kitchen_df = pd.read_csv(kitchen_file)
        kitchen_df['TaskType'] = 'kitchen'
        all_results.append(kitchen_df)
        print(f"成功读取Kitchen结果: {kitchen_file}")
    except Exception as e:
        print(f"读取Kitchen结果文件出错: {e}")

# 如果无法从CSV读取数据，尝试直接从结果目录读取评估结果
if not all_results:
    print("未找到有效的结果文件，尝试从结果目录直接读取数据...")
    
    # 定义任务类型和对应的任务
    tasks = {
        'ph': ['lift_ph', 'can_ph', 'square_ph', 'transport_ph', 'tool_hang_ph', 'pusht'],
        'mh': ['lift_mh', 'can_mh', 'square_mh', 'transport_mh'],
        'bp': ['block_pushing'],
        'kitchen': ['kitchen']
    }
    
    # 手动构建结果数据
    manual_results = []
    
    # 遍历所有任务目录
    for task_type, task_list in tasks.items():
        for task in task_list:
            task_dir = os.path.join(result_dir, task)
            
            if not os.path.exists(task_dir):
                continue
                
            # 查找所有种子和检查点类型的目录
            for seed_dir in glob.glob(os.path.join(task_dir, "seed*")):
                # 提取种子和检查点类型
                dir_name = os.path.basename(seed_dir)
                parts = dir_name.split('_')
                if len(parts) < 2:
                    continue
                    
                seed = parts[0].replace('seed', '')
                checkpoint_type = parts[1]
                
                # 读取评估结果
                result_file = os.path.join(seed_dir, "fast_eval_log.json")
                metrics_file = os.path.join(seed_dir, "eval_results.json")
                
                if not os.path.exists(result_file):
                    continue
                
                # 读取结果
                try:
                    # 对于block_pushing任务，处理p1和p2
                    if task == 'block_pushing':
                        # 尝试从JSON读取
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                                bp_p1 = result_data.get('test/p1', 0.0)
                                bp_p2 = result_data.get('test/p2', 0.0)
                        except:
                            # 如果JSON解析失败，尝试直接grep
                            with open(result_file, 'r') as f:
                                content = f.read()
                                import re
                                bp_p1_match = re.search(r'"test/p1":\s*([\d\.]+)', content)
                                bp_p2_match = re.search(r'"test/p2":\s*([\d\.]+)', content)
                                bp_p1 = float(bp_p1_match.group(1)) if bp_p1_match else 0.0
                                bp_p2 = float(bp_p2_match.group(1)) if bp_p2_match else 0.0
                        
                        # 读取speedup和flops
                        speedup = "-"
                        flops = "-"
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics_data = json.load(f)
                                    speedup = metrics_data.get('speedup', '-')
                                    flops = metrics_data.get('flops', '-')
                            except:
                                with open(metrics_file, 'r') as f:
                                    content = f.read()
                                    speedup_match = re.search(r'"speedup":\s*([\d\.]+)', content)
                                    flops_match = re.search(r'"flops":\s*([\d\.]+)', content)
                                    speedup = float(speedup_match.group(1)) if speedup_match else "-"
                                    flops = float(flops_match.group(1)) if flops_match else "-"
                        
                        # 添加p1和p2的结果
                        manual_results.append({
                            'Method': 'BAC',
                            'Steps': ${STEPS},
                            'Task': f'{task}_p1',
                            'Seed': seed,
                            'CheckpointType': checkpoint_type,
                            'SuccessRate': bp_p1,
                            'FLOPs': flops,
                            'Speedup': speedup,
                            'TaskType': task_type
                        })
                        
                        manual_results.append({
                            'Method': 'BAC',
                            'Steps': ${STEPS},
                            'Task': f'{task}_p2',
                            'Seed': seed,
                            'CheckpointType': checkpoint_type,
                            'SuccessRate': bp_p2,
                            'FLOPs': flops,
                            'Speedup': speedup,
                            'TaskType': task_type
                        })
                        
                    elif task == 'kitchen':
                        # 尝试从JSON读取
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                                kitchen_p1 = result_data.get('test/p_1', 0.0)
                                kitchen_p2 = result_data.get('test/p_2', 0.0)
                                kitchen_p3 = result_data.get('test/p_3', 0.0)
                                kitchen_p4 = result_data.get('test/p_4', 0.0)
                        except:
                            # 如果JSON解析失败，尝试直接grep
                            with open(result_file, 'r') as f:
                                content = f.read()
                                import re
                                p1_match = re.search(r'"test/p_1":\s*([\d\.]+)', content)
                                p2_match = re.search(r'"test/p_2":\s*([\d\.]+)', content)
                                p3_match = re.search(r'"test/p_3":\s*([\d\.]+)', content)
                                p4_match = re.search(r'"test/p_4":\s*([\d\.]+)', content)
                                kitchen_p1 = float(p1_match.group(1)) if p1_match else 0.0
                                kitchen_p2 = float(p2_match.group(1)) if p2_match else 0.0
                                kitchen_p3 = float(p3_match.group(1)) if p3_match else 0.0
                                kitchen_p4 = float(p4_match.group(1)) if p4_match else 0.0
                        
                        # 读取speedup和flops
                        speedup = "-"
                        flops = "-"
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics_data = json.load(f)
                                    speedup = metrics_data.get('speedup', '-')
                                    flops = metrics_data.get('flops', '-')
                            except:
                                with open(metrics_file, 'r') as f:
                                    content = f.read()
                                    speedup_match = re.search(r'"speedup":\s*([\d\.]+)', content)
                                    flops_match = re.search(r'"flops":\s*([\d\.]+)', content)
                                    speedup = float(speedup_match.group(1)) if speedup_match else "-"
                                    flops = float(flops_match.group(1)) if flops_match else "-"
                        
                        # 添加p1-p4的结果
                        for idx, p_value in enumerate([kitchen_p1, kitchen_p2, kitchen_p3, kitchen_p4], 1):
                            manual_results.append({
                                'Method': 'BAC',
                                'Steps': ${STEPS},
                                'Task': f'{task}_p{idx}',
                                'Seed': seed,
                                'CheckpointType': checkpoint_type,
                                'SuccessRate': p_value,
                                'FLOPs': flops,
                                'Speedup': speedup,
                                'TaskType': task_type
                            })
                    else:
                        # 普通任务
                        # 尝试从metrics文件读取成功率
                        success_rate = 0.0
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics_data = json.load(f)
                                    success_rate = metrics_data.get('mean_score', 0.0)
                                    speedup = metrics_data.get('speedup', '-')
                                    flops = metrics_data.get('flops', '-')
                            except:
                                with open(metrics_file, 'r') as f:
                                    content = f.read()
                                    success_match = re.search(r'"mean_score":\s*([\d\.]+)', content)
                                    speedup_match = re.search(r'"speedup":\s*([\d\.]+)', content)
                                    flops_match = re.search(r'"flops":\s*([\d\.]+)', content)
                                    success_rate = float(success_match.group(1)) if success_match else 0.0
                                    speedup = float(speedup_match.group(1)) if speedup_match else "-"
                                    flops = float(flops_match.group(1)) if flops_match else "-"
                        
                        manual_results.append({
                            'Method': 'BAC',
                            'Steps': ${STEPS},
                            'Task': task,
                            'Seed': seed,
                            'CheckpointType': checkpoint_type,
                            'SuccessRate': success_rate,
                            'FLOPs': flops,
                            'Speedup': speedup,
                            'TaskType': task_type
                        })
                except Exception as e:
                    print(f"处理任务 {task} (种子 {seed}, 检查点类型 {checkpoint_type}) 时出错: {e}")
    
    # 如果成功构建了结果，将其转换为DataFrame
    if manual_results:
        results_df = pd.DataFrame(manual_results)
        print(f"成功从结果目录手动构建了 {len(manual_results)} 条结果数据")
    else:
        print("无法从结果目录构建数据，退出")
        exit(1)
else:
    # 使用从CSV读取的数据
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"从CSV文件合并后数据形状: {results_df.shape}")

# 美化任务名称(用于表格显示)
task_display_names = {
    'lift_ph': 'Lift$_{ph}$', 
    'can_ph': 'Can$_{ph}$', 
    'square_ph': 'Square$_{ph}$',
    'transport_ph': 'Trans$_{ph}$', 
    'tool_hang_ph': 'Tool$_{ph}$', 
    'pusht': 'Push--T',
    'lift_mh': 'Lift$_{mh}$', 
    'can_mh': 'Can$_{mh}$', 
    'square_mh': 'Square$_{mh}$', 
    'transport_mh': 'Trans$_{mh}$',
    'block_pushing_p1': 'BP$_{p1}$', 
    'block_pushing_p2': 'BP$_{p2}$',
    'kitchen_p1': 'Kit$_{p1}$', 
    'kitchen_p2': 'Kit$_{p2}$', 
    'kitchen_p3': 'Kit$_{p3}$', 
    'kitchen_p4': 'Kit$_{p4}$'
}

# 显示部分数据以便调试
print("\n数据示例:")
print(results_df[['Method', 'Steps', 'Task', 'SuccessRate', 'Speedup', 'CheckpointType']].head(10))

# 生成汇总数据
summary_data = []

# 检查DataFrame
print(f"\n处理数据的列名: {list(results_df.columns)}")
print(f"方法列中的值: {results_df['Method'].unique()}")
print(f"步数列中的值: {results_df['Steps'].unique()}")
print(f"任务列中的值: {results_df['Task'].unique()}")

# 按任务类型、方法、步数、任务名称分组
grouped = results_df.groupby(['TaskType', 'Method', 'Steps'])
print(f"分组后的组数: {len(grouped)}")

for (task_type, method, steps), group in grouped:
    print(f"处理组: {task_type}, {method}, {steps}, 数据量: {len(group)}")
    
    for task in sorted(group['Task'].unique()):
        # 计算最大性能和平均性能
        task_data = group[group['Task'] == task]
        
        if task_data.empty:
            print(f"  - 跳过空组: {task}")
            continue
            
        # 处理检查点类型问题
        if 'CheckpointType' in task_data.columns:
            max_success = task_data[task_data['CheckpointType'] == 'max']['SuccessRate'].mean()
            avg_success = task_data[task_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
        else:
            # 如果没有CheckpointType列，则直接使用SuccessRate
            print(f"  - 警告: {task} 没有CheckpointType列，使用直接平均")
            max_success = avg_success = task_data['SuccessRate'].mean()
        
        # 计算FLOPs和加速比
        if 'FLOPs' in task_data.columns:
            flops = task_data['FLOPs'].mean() if not task_data['FLOPs'].isnull().all() else float('nan')
        else:
            flops = float('nan')
            
        if 'Speedup' in task_data.columns:
            speedup = task_data['Speedup'].mean() if not task_data['Speedup'].isnull().all() else float('nan')
        else:
            speedup = float('nan')
        
        # 添加到汇总数据
        summary_data.append({
            'TaskType': task_type,
            'Method': method,
            'Steps': steps,
            'Task': task,
            'MaxSuccessRate': max_success,
            'AvgSuccessRate': avg_success,
            'FLOPs': flops,
            'Speedup': speedup
        })
        print(f"  - 添加: {task}, 成功率: {max_success:.3f}/{avg_success:.3f}")

# 创建汇总DataFrame
summary_df = pd.DataFrame(summary_data)
print(f"汇总数据形状: {summary_df.shape}")

# 填充缺失值为"-"
summary_df = summary_df.fillna('-')

# 删除原来的表格数据输出部分，只保留表格格式汇总数据
print("\n\033[1;34m===== 表格格式汇总数据 =====\033[0m")

# 在Python代码中定义任务列表
ph_tasks = ['lift_ph', 'can_ph', 'square_ph', 'transport_ph', 'tool_hang_ph', 'pusht']
mh_tasks = ['lift_mh', 'can_mh', 'square_mh', 'transport_mh']
bp_tasks = ['block_pushing_p1', 'block_pushing_p2']
kitchen_tasks = ['kitchen_p1', 'kitchen_p2', 'kitchen_p3', 'kitchen_p4']
lowdim_tasks = bp_tasks + kitchen_tasks

# PH任务表格行
print("\n\033[1;36mPH任务表格行 (BAC):\033[0m")
for steps in sorted(summary_df['Steps'].unique()):
    # 找出当前步数下所有PH任务的数据
    ph_task_data = {}
    for task in ph_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            max_sr = task_rows['MaxSuccessRate'].iloc[0]
            avg_sr = task_rows['AvgSuccessRate'].iloc[0]
            
            # 防止"nan"和非数字值
            if max_sr != '-' and not pd.isna(max_sr):
                max_sr_str = f"{float(max_sr):.2f}"
            else:
                max_sr_str = "--"
                
            if avg_sr != '-' and not pd.isna(avg_sr):
                avg_sr_str = f"{float(avg_sr):.2f}"
            else:
                avg_sr_str = "--"
                
            ph_task_data[task] = f"{max_sr_str}/{avg_sr_str}"
        else:
            ph_task_data[task] = "--/--"
    
    # 如果没有找到任何数据，跳过
    if not ph_task_data:
        continue
    
    # 获取加速比
    speedup_values = []
    for task in ph_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            speedup = task_rows['Speedup'].iloc[0]
            if speedup != '-' and not pd.isna(speedup):
                try:
                    speedup_values.append(float(speedup))
                except:
                    pass
    
    # 计算平均加速比
    if speedup_values:
        avg_speedup = sum(speedup_values) / len(speedup_values)
        speedup_str = f"{avg_speedup:.2f}"
    else:
        speedup_str = "--"
    
    # 生成表格行
    ph_row = f"BAC & {steps} & {ph_task_data.get('lift_ph', '--/--')} & {ph_task_data.get('can_ph', '--/--')} & {ph_task_data.get('square_ph', '--/--')} & {ph_task_data.get('transport_ph', '--/--')} & {ph_task_data.get('tool_hang_ph', '--/--')} & {ph_task_data.get('pusht', '--/--')} & -- & {speedup_str}"
    print(ph_row)

# MH任务表格行
print("\n\033[1;36mMH任务表格行 (BAC):\033[0m")
for steps in sorted(summary_df['Steps'].unique()):
    # 找出当前步数下所有MH任务的数据
    mh_task_data = {}
    for task in mh_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            max_sr = task_rows['MaxSuccessRate'].iloc[0]
            avg_sr = task_rows['AvgSuccessRate'].iloc[0]
            
            # 防止"nan"和非数字值
            if max_sr != '-' and not pd.isna(max_sr):
                max_sr_str = f"{float(max_sr):.2f}"
            else:
                max_sr_str = "--"
                
            if avg_sr != '-' and not pd.isna(avg_sr):
                avg_sr_str = f"{float(avg_sr):.2f}"
            else:
                avg_sr_str = "--"
                
            mh_task_data[task] = f"{max_sr_str}/{avg_sr_str}"
        else:
            mh_task_data[task] = "--/--"
    
    # 如果没有找到任何数据，跳过
    if not mh_task_data:
        continue
    
    # 获取加速比
    speedup_values = []
    for task in mh_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            speedup = task_rows['Speedup'].iloc[0]
            if speedup != '-' and not pd.isna(speedup):
                try:
                    speedup_values.append(float(speedup))
                except:
                    pass
    
    # 计算平均加速比
    if speedup_values:
        avg_speedup = sum(speedup_values) / len(speedup_values)
        speedup_str = f"{avg_speedup:.2f}"
    else:
        speedup_str = "--"
    
    # 生成表格行
    mh_row = f"BAC & {steps} & {mh_task_data.get('lift_mh', '--/--')} & {mh_task_data.get('can_mh', '--/--')} & {mh_task_data.get('square_mh', '--/--')} & {mh_task_data.get('transport_mh', '--/--')} & -- & {speedup_str}"
    print(mh_row)

# LOWDIM任务表格行
print("\n\033[1;36mLOWDIM任务表格行 (BAC):\033[0m")
for steps in sorted(summary_df['Steps'].unique()):
    # 找出当前步数下所有LOWDIM任务的数据
    lowdim_task_data = {}
    for task in lowdim_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            max_sr = task_rows['MaxSuccessRate'].iloc[0]
            avg_sr = task_rows['AvgSuccessRate'].iloc[0]
            
            # 防止"nan"和非数字值
            if max_sr != '-' and not pd.isna(max_sr):
                max_sr_str = f"{float(max_sr):.2f}"
            else:
                max_sr_str = "--"
                
            if avg_sr != '-' and not pd.isna(avg_sr):
                avg_sr_str = f"{float(avg_sr):.2f}"
            else:
                avg_sr_str = "--"
                
            lowdim_task_data[task] = f"{max_sr_str}/{avg_sr_str}"
        else:
            lowdim_task_data[task] = "--/--"
    
    # 如果没有找到任何数据，跳过
    if not lowdim_task_data:
        continue
    
    # 获取加速比
    speedup_values = []
    for task in lowdim_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            speedup = task_rows['Speedup'].iloc[0]
            if speedup != '-' and not pd.isna(speedup):
                try:
                    speedup_values.append(float(speedup))
                except:
                    pass
    
    # 计算平均加速比
    if speedup_values:
        avg_speedup = sum(speedup_values) / len(speedup_values)
        speedup_str = f"{avg_speedup:.2f}"
    else:
        speedup_str = "--"
    
    # 生成表格行 - 使用正确的lowdim任务列表
    lowdim_row = f"BAC & {steps} & {lowdim_task_data.get('block_pushing_p1', '--/--')} & {lowdim_task_data.get('block_pushing_p2', '--/--')} & {lowdim_task_data.get('kitchen_p1', '--/--')} & {lowdim_task_data.get('kitchen_p2', '--/--')} & {lowdim_task_data.get('kitchen_p3', '--/--')} & {lowdim_task_data.get('kitchen_p4', '--/--')} & -- & {speedup_str}"
    print(lowdim_row)

# 保存汇总文件
summary_file = os.path.join(result_dir, "summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"\n汇总数据已保存至: {summary_file}")
print(f"汇总数据形状: {summary_df.shape}")
EOF

print_title "BAC模型评估完成"
print_info "最终结果文件:"
print_info "- 主结果文件: $RESULTS_FILE"
print_info "- 汇总结果文件: $SUMMARY_FILE" 