#!/bin/bash

# 消融实验脚本：比较Unified ACS和Block-wise ACS
# Unified ACS: 使用edit模式，所有块共用一组自适应步骤
# Block-wise ACS: 使用optimal模式且num_bu_blocks=0，每个块使用独立的自适应步骤
#
# bu参数说明:
# num_bu_blocks - block update的块数，值为0表示对每个块独立使用自适应步骤
#               - 值为大于0的整数表示使用block update机制，将模型分为指定数量的块
#
# 输出目录格式:
# results/${TASK_NAME}/${CACHE_MODE}/${METRIC}_caches${NUM_CACHE}_bu${NUM_BU_BLOCKS}_seed${SEED}_${CHECKPOINT_TYPE}

DEVICE="cuda:0"
SKIP_VIDEO="--skip_video"
METRIC="cosine"
TASK_TYPE="both" # 默认同时测试PH和MH任务，可选值: "ph", "mh", "both", "lowdim", "all"
FORCE_RECOMPUTE=false # 是否强制重新计算，即使已有结果文件

# 定义不同步数配置
declare -a STEPS_CONFIGS=(10)

定义任务列表
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

# 添加低维任务列表
declare -a LOWDIM_TASKS=(
    "block_pushing"
    "kitchen"
)

# 当前活跃的任务列表（将在命令行参数解析后根据TASK_TYPE设置）
declare -a ACTIVE_TASKS=()

# 定义任务和对应的检查点路径 (最大性能)
declare -A MAX_CHECKPOINTS_SEED0=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0950-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2400-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2100-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.773.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.748.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=3050-test_mean_score=1.000.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
    # 添加LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED1=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1150-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2200-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1800-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=1100-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0200-test_mean_score=0.773.ckpt"
    # 添加LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/epoch=2700-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED2=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3350-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=3050-test_mean_score=0.955.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0350-test_mean_score=0.682.ckpt"
    # 添加LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/epoch=1750-test_mean_score=0.574.ckpt"
)

# 定义任务和对应的检查点路径 (最后10个检查点平均)
declare -A AVG_CHECKPOINTS_SEED0=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    # 添加LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED1=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    # 添加LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED2=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    # 添加LOWDIM任务
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)


# 定义种子列表
declare -a SEEDS=(0 1 2)

# 使用说明 (定义在此处，以便在参数解析错误时调用)
usage() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --device device_name            指定要使用的设备 (默认: ${DEVICE})"
    echo "  --task_type type                指定要评估的任务类型 (ph, mh, both, lowdim, all) (默认: both)"
    echo "  --ph_tasks task1,task2,...      指定要评估的PH任务 (默认: 所有PH任务)"
    echo "  --mh_tasks task1,task2,...      指定要评估的MH任务 (默认: 所有MH任务)"
    echo "  --lowdim_tasks task1,task2,...  指定要评估的低维任务 (默认: 所有低维任务)"
    echo "  --steps steps1,steps2,...       指定要评估的步数 (默认: 10)"
    echo "  --seeds seed1,seed2,...         指定要评估的种子 (默认: 0,1,2)"
    echo "  --metric metric                 指定要使用的度量标准 (默认: cosine)"
    echo "  --skip_video                    跳过视频渲染"
    echo "  --force                         强制重新计算，即使结果已存在"
    echo "  --bp_only                       仅评估block_pushing任务"
    echo "  --kitchen_only                  仅评估kitchen任务"
    echo
    echo "说明:"
    echo "  本脚本评估两种自适应缓存调度方法的性能:"
    echo "  1. Unified ACS: 使用edit模式，所有块共用一组自适应步骤"
    echo "  2. Block-wise ACS: 使用optimal模式且num_bu_blocks=0，每个块使用独立的自适应步骤"
    echo
    echo "示例:"
    echo "  $0 --device cuda:1 --task_type ph --ph_tasks lift_ph,can_ph --steps 5 --seeds 0 --skip_video"
    echo "  $0 --device cuda:0 --task_type both --steps 5,10 --metric cosine"
    echo "  $0 --device cuda:0 --task_type mh --mh_tasks lift_mh,can_mh --force"
    echo "  $0 --device cuda:0 --task_type lowdim --steps 5 --seeds 0 --skip_video"
    echo "  $0 --bp_only --device cuda:0 --steps 10 --metric mse"
}

# 解析命令行参数覆盖默认值
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
        --ph_tasks)
            IFS=',' read -ra PH_TASKS <<< "$2"
            shift 2
            ;;
        --mh_tasks)
            IFS=',' read -ra MH_TASKS <<< "$2"
            shift 2
            ;;
        --lowdim_tasks)
            IFS=',' read -ra LOWDIM_TASKS <<< "$2"
            shift 2
            ;;
        --steps)
            IFS=',' read -ra STEPS_CONFIGS <<< "$2"
            shift 2
            ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --skip_video)
            SKIP_VIDEO="--skip_video"
            shift
            ;;
        --force)
            FORCE_RECOMPUTE=true
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

# 根据task_type设置活跃任务列表
case "$TASK_TYPE" in
    "ph")
        ACTIVE_TASKS=("${PH_TASKS[@]}")
        ;;
    "mh")
        ACTIVE_TASKS=("${MH_TASKS[@]}")
        ;;
    "both")
        ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}")
        ;;
    "lowdim")
        ACTIVE_TASKS=("${LOWDIM_TASKS[@]}")
        ;;
    "all")
        ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}" "${LOWDIM_TASKS[@]}")
        ;;
    "block_pushing")
        ACTIVE_TASKS=("block_pushing")
        ;;
    "kitchen")
        ACTIVE_TASKS=("kitchen")
        ;;
    *)
        echo "错误: 无效的任务类型 '$TASK_TYPE', 必须是 'ph', 'mh', 'both', 'lowdim', 'all', 'block_pushing' 或 'kitchen'"
        exit 1
        ;;
esac

# 创建结果目录
BENCHMARK_RESULTS_DIR="results/ablation/bac"
mkdir -p "$BENCHMARK_RESULTS_DIR"

# 设置全局输出目录基础
OUTPUT_DIR_BASE="results/ablation/bac"
mkdir -p "$OUTPUT_DIR_BASE"

# 创建结果文件
RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/results.csv"
PH_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/ph_results.csv"
MH_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/mh_results.csv"
BP_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/bp_results.csv"
KITCHEN_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/kitchen_results.csv"

# 初始化结果文件
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup,TaskType" > "$RESULTS_FILE"

# 根据任务类型创建相应的结果文件
if [[ "$TASK_TYPE" == "ph" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$PH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "mh" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$MH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "block_pushing" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$BP_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "kitchen" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$KITCHEN_RESULTS_FILE"
fi

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

# 执行单次评估的函数
# 如果结果文件已存在且不是强制重新计算模式，则直接读取结果
execute_single_evaluation() {
    local method_name="$1"
    local cache_mode="$2"
    local cache_threshold="$3"  # 对threshold模式有效
    local edit_steps="$4"       # 对edit模式有效
    local optimal_steps_dir="$5" # 对optimal模式有效
    local num_caches="$6"        # 对optimal模式有效
    local num_bu_blocks="$7"     # 对optimal模式有效
    local task_name="$8"
    local task_type_name="$9"
    local seed="${10}"
    local checkpoint_type="${11}"
    local checkpoint="${12}"

    # 为当前评估设置全局输出目录
    local cache_mode_var="${cache_mode}"
    local num_cache_var="${num_caches}"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${method_name}/${cache_mode_var}_${METRIC}_caches${num_cache_var}_bu${num_bu_blocks}"
    mkdir -p "$OUTPUT_DIR"

    # 构建任务特定的输出目录
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    mkdir -p "$task_output_dir"
    local result_file="${task_output_dir}/fast_eval_log.json"
    local metrics_file="${task_output_dir}/eval_results.json"

    # 检查是否已有结果文件，且不是强制重新计算模式
    if [ -f "${result_file}" ] && [ -f "${metrics_file}" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_info "发现已存在的结果文件，直接读取: ${task_output_dir}"

        # 从现有文件中读取结果
        # 处理block_pushing任务的特殊结果格式
        if [ "$task_name" = "block_pushing" ]; then
            # 提取p1和p2的成功率
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 记录结果到主结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"

            # 记录结果到Block-pushing特定结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup}" >> "$BP_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup}" >> "$BP_RESULTS_FILE"

            print_info "从现有文件读取结果成功:"
            echo "  BP P1: ${bp_p1}"
            echo "  BP P2: ${bp_p2}"
            echo "  FLOPs: ${flops}"
            echo "  Speedup: ${speedup}"

            print_separator
            return 0
        # 处理kitchen任务的特殊结果格式
        elif [ "$task_name" = "kitchen" ]; then
            # 提取p_1到p_4的成功率
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 记录结果到主结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},kitchen" >> "$RESULTS_FILE"

            # 记录结果到Kitchen特定结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"

            print_info "从现有文件读取结果成功:"
            echo "  Kitchen P1: ${kitchen_p1}"
            echo "  Kitchen P2: ${kitchen_p2}"
            echo "  Kitchen P3: ${kitchen_p3}"
            echo "  Kitchen P4: ${kitchen_p4}"
            echo "  FLOPs: ${flops}"
            echo "  Speedup: ${speedup}"

            print_separator
            return 0
        # 处理一般任务的结果格式
        else
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1)
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 检查是否成功读取到有效结果
            if [ -n "$success_rate" ] && [ "$success_rate" != "N/A" ]; then
                print_info "从现有文件读取结果成功:"
                echo "  Success Rate: ${success_rate}"
                echo "  FLOPs: ${flops}"
                echo "  Speedup: ${speedup}"

                # 记录结果到主结果文件
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type_name}" >> "$RESULTS_FILE"

                # 根据任务类型记录到对应结果文件
                if [ "$task_type_name" == "ph" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$PH_RESULTS_FILE"
                elif [ "$task_type_name" == "mh" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$MH_RESULTS_FILE"
                elif [ "$task_type_name" == "block_pushing" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$BP_RESULTS_FILE"
                elif [ "$task_type_name" == "kitchen" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
                fi

                print_separator
                return 0
            else
                print_warning "现有结果文件中未找到有效数据，将重新运行评估"
            fi
        fi
    elif [ "$FORCE_RECOMPUTE" = true ]; then
        print_info "强制重新计算模式：将重新运行评估"
    fi

    # 打印评估信息
    print_info "评估: ${task_name} (${task_type_name})"
    if [ "$cache_mode" == "threshold" ]; then
        echo "模式: ${cache_mode}, 步数: ${num_caches}, 阈值: ${cache_threshold}"
    elif [ "$cache_mode" == "edit" ]; then
        echo "模式: ${cache_mode}, 步数: ${num_caches}"
    elif [ "$cache_mode" == "optimal" ]; then
        echo "模式: ${cache_mode}, 步数: ${num_caches}, BU块数: ${num_bu_blocks}"
    else
        echo "模式: ${cache_mode}, 步数: ${num_caches}"
    fi
    echo "种子: ${seed}, 检查点类型: ${checkpoint_type}"
    echo "检查点: ${checkpoint}"
    echo "输出目录: ${task_output_dir}"

    # 构建评估命令的参数
    local eval_args=(
        --checkpoint "${checkpoint}"
        --output_dir "${task_output_dir}"
        --device "${DEVICE}"
        --cache_mode "${cache_mode}"
    )

    # 根据缓存模式添加不同参数
    if [ "$cache_mode" == "threshold" ] && [ -n "$cache_threshold" ]; then
        eval_args+=(--cache_threshold "${cache_threshold}")
    elif [ "$cache_mode" == "edit" ] && [ -n "$edit_steps" ]; then
        eval_args+=(--edit_steps "${edit_steps}")
    elif [ "$cache_mode" == "optimal" ] && [ -n "$optimal_steps_dir" ]; then
        eval_args+=(--optimal_steps_dir "${optimal_steps_dir}")
        eval_args+=(--num_caches "${num_caches}")
        eval_args+=(--metric "${METRIC}")
        eval_args+=(--num_bu_blocks "${num_bu_blocks}")
    fi

    # 添加跳过视频参数
    if [ -n "$SKIP_VIDEO" ]; then
        eval_args+=($SKIP_VIDEO)
    fi

    # 执行评估
    python scripts/eval_fast_diffusion_policy.py "${eval_args[@]}"

    # 检查结果文件是否生成
    if [ -f "${result_file}" ]; then
        # 处理block_pushing任务的特殊结果格式
        if [ "$task_name" = "block_pushing" ]; then
            # 提取p1和p2的成功率
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 记录结果到主结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"

            # 记录结果到Block-pushing特定结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup}" >> "$BP_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup}" >> "$BP_RESULTS_FILE"

            print_info "评估完成: ${task_name} p1=${bp_p1}, p2=${bp_p2} (${method_name}, 步数: ${num_caches}, 种子: ${seed}, 检查点类型: ${checkpoint_type})"
        # 处理kitchen任务的特殊结果格式
        elif [ "$task_name" = "kitchen" ]; then
            # 提取p_1到p_4的成功率
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 记录结果到主结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},kitchen" >> "$RESULTS_FILE"

            # 记录结果到Kitchen特定结果文件
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"

            print_info "评估完成: ${task_name} p1=${kitchen_p1}, p2=${kitchen_p2}, p3=${kitchen_p3}, p4=${kitchen_p4} (${method_name}, 步数: ${num_caches}, 种子: ${seed}, 检查点类型: ${checkpoint_type})"
        # 处理一般任务的结果格式
        else
            # 从结果文件中提取成功率和加速比
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1)
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 记录结果到主结果文件
            echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type_name}" >> "$RESULTS_FILE"

            # 根据任务类型记录到对应结果文件
            if [ "$task_type_name" == "ph" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$PH_RESULTS_FILE"
            elif [ "$task_type_name" == "mh" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$MH_RESULTS_FILE"
            elif [ "$task_type_name" == "block_pushing" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$BP_RESULTS_FILE"
            elif [ "$task_type_name" == "kitchen" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            fi

            print_info "评估完成: 成功率=${success_rate}, 加速比=${speedup}"
        fi
    else
        print_warning "警告: 结果文件不存在: ${result_file}"
    fi

    print_separator
}

# 打印实验配置信息
print_title "====================================================="
print_title "               BAC消融实验: Unified vs Block-wise     "
print_title "====================================================="
print_info "设备: $DEVICE"
print_info "评估步数: ${STEPS_CONFIGS[*]}"
print_info "任务类型: $TASK_TYPE"
print_info "活跃任务: ${ACTIVE_TASKS[*]}"
print_info "种子: ${SEEDS[*]}"
print_info "指标: $METRIC"
print_info "跳过视频: ${SKIP_VIDEO:+是}"
print_info "强制重新计算: ${FORCE_RECOMPUTE}"
print_title "====================================================="
echo ""

# 运行评估
for STEPS in "${STEPS_CONFIGS[@]}"; do
    for TASK_NAME in "${ACTIVE_TASKS[@]}"; do
        # 获取任务类型
        if [[ "$TASK_NAME" == *"_ph"* ]] || [[ "$TASK_NAME" == "pusht" ]]; then
            TASK_TYPE_NAME="ph"
        elif [[ "$TASK_NAME" == *"_mh"* ]]; then
            TASK_TYPE_NAME="mh"
        elif [[ "$TASK_NAME" == "block_pushing" ]]; then
            TASK_TYPE_NAME="block_pushing"
        elif [[ "$TASK_NAME" == "kitchen" ]]; then
            TASK_TYPE_NAME="kitchen"
        else
            TASK_TYPE_NAME="unknown"
        fi

        print_subtitle "处理任务: ${TASK_NAME} (${TASK_TYPE_NAME}), 步数: ${STEPS}"

        # 准备评估所需的路径和文件
        OPTIMAL_STEPS_DIR="assets/${TASK_NAME}/original/optimal_steps/${METRIC}"
        OPTIMAL_STEPS_FILE="${OPTIMAL_STEPS_DIR}/decoder.layers.0.self_attn/optimal_steps_decoder.layers.0.self_attn_${STEPS}_${METRIC}.pkl"
        EDIT_STEPS=""

        # 读取最优步骤并格式化为逗号分隔的字符串（如果文件存在）
        if [ -f "$OPTIMAL_STEPS_FILE" ]; then
            # 从pkl文件提取步骤
            EDIT_STEPS=$(python -c "
import pickle
with open('${OPTIMAL_STEPS_FILE}', 'rb') as f:
    steps = pickle.load(f)
print(','.join(map(str, steps)))
")
            print_info "从文件加载的最优步骤: $EDIT_STEPS"
        else
            print_warning "未找到最优步骤文件: $OPTIMAL_STEPS_FILE"
            print_error "执行失败"
            print_separator
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            # 获取对应种子的检查点路径
            if [ "$SEED" == "0" ]; then
                MAX_CHECKPOINT="${MAX_CHECKPOINTS_SEED0[$TASK_NAME]}"
                AVG_CHECKPOINT="${AVG_CHECKPOINTS_SEED0[$TASK_NAME]}"
            elif [ "$SEED" == "1" ]; then
                MAX_CHECKPOINT="${MAX_CHECKPOINTS_SEED1[$TASK_NAME]}"
                AVG_CHECKPOINT="${AVG_CHECKPOINTS_SEED1[$TASK_NAME]}"
            else
                MAX_CHECKPOINT="${MAX_CHECKPOINTS_SEED2[$TASK_NAME]}"
                AVG_CHECKPOINT="${AVG_CHECKPOINTS_SEED2[$TASK_NAME]}"
            fi

            # 为每种检查点类型运行评估
            for CHECKPOINT_TYPE in "max" "avg"; do
                if [ "$CHECKPOINT_TYPE" == "max" ]; then
                    CHECKPOINT="$MAX_CHECKPOINT"
                else
                    CHECKPOINT="$AVG_CHECKPOINT"
                fi

                # 检查检查点文件是否存在
                if [ ! -f "$CHECKPOINT" ]; then
                    print_warning "警告: 检查点文件不存在: $CHECKPOINT, 跳过评估"
                    continue
                fi

                # 执行Unified ACS评估 (edit模式)
                execute_single_evaluation "Unified ACS" "edit" "" "$EDIT_STEPS" "" \
                    "$STEPS" "" "$TASK_NAME" "$TASK_TYPE_NAME" "$SEED" "$CHECKPOINT_TYPE" "$CHECKPOINT"

                # 执行Block-wise ACS评估 (optimal模式，num_bu_blocks=0)
                execute_single_evaluation "Block-wise ACS" "optimal" "" "" "$OPTIMAL_STEPS_DIR" \
                    "$STEPS" "0" "$TASK_NAME" "$TASK_TYPE_NAME" "$SEED" "$CHECKPOINT_TYPE" "$CHECKPOINT"
            done
        done
    done
done

# 汇总结果
print_title "汇总结果..."
python - <<EOF
import pandas as pd
import numpy as np
# 读取结果文件
results = pd.read_csv("$RESULTS_FILE")
# 检查是否有各类任务的结果
has_ph = any(results["TaskType"] == "ph")
has_mh = any(results["TaskType"] == "mh")
has_bp = any(results["TaskType"] == "block_pushing")
has_kitchen = any(results["TaskType"] == "kitchen")
# 处理PH任务表格数据
if has_ph:
    print("\n\033[1;34m===== PH任务 BAC消融实验表格数据 =====\033[0m")
    ph_results = results[results["TaskType"] == "ph"]
    
    # 生成LaTeX表格格式的结果
    for method in sorted(ph_results["Method"].unique()):
        for steps in sorted(ph_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # 处理每个任务
            ph_tasks = sorted(list(set([task for task in ph_results["Task"].unique() if task.endswith("_ph") or task == "pusht"])))
            for task in ph_tasks:
                # 筛选数据
                task_data = ph_results[(ph_results["Method"] == method) & 
                                     (ph_results["Steps"] == steps) & 
                                     (ph_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 计算最大性能和平均性能
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # 计算FLOPs和加速比
            flops = ph_results[(ph_results["Method"] == method) & 
                             (ph_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = ph_results[(ph_results["Method"] == method) & 
                             (ph_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 打印各方法的平均性能
    print("\n\033[1;34m----- PH任务方法性能比较 -----\033[0m")
    for method in sorted(ph_results["Method"].unique()):
        method_data = ph_results[ph_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}: 平均成功率 = {avg_success:.3f}, 平均加速比 = {avg_speedup:.2f}x")
# 处理MH任务表格数据
if has_mh:
    print("\n\033[1;34m===== MH任务 BAC消融实验表格数据 =====\033[0m")
    mh_results = results[results["TaskType"] == "mh"]
    
    # 生成LaTeX表格格式的结果
    for method in sorted(mh_results["Method"].unique()):
        for steps in sorted(mh_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # 处理每个任务
            mh_tasks = sorted(list(set([task for task in mh_results["Task"].unique() if task.endswith("_mh")])))
            for task in mh_tasks:
                # 筛选数据
                task_data = mh_results[(mh_results["Method"] == method) & 
                                     (mh_results["Steps"] == steps) & 
                                     (mh_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 计算最大性能和平均性能
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # 计算FLOPs和加速比
            flops = mh_results[(mh_results["Method"] == method) & 
                             (mh_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = mh_results[(mh_results["Method"] == method) & 
                             (mh_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 打印各方法的平均性能
    print("\n\033[1;34m----- MH任务方法性能比较 -----\033[0m")
    for method in sorted(mh_results["Method"].unique()):
        method_data = mh_results[mh_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}: 平均成功率 = {avg_success:.3f}, 平均加速比 = {avg_speedup:.2f}x")
# 处理Block-pushing任务表格数据
if has_bp:
    print("\n\033[1;34m===== Block-pushing任务 BAC消融实验表格数据 =====\033[0m")
    bp_results = results[results["TaskType"] == "block_pushing"]
    
    # 生成LaTeX表格格式的结果
    for method in sorted(bp_results["Method"].unique()):
        for steps in sorted(bp_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # 处理p1和p2任务
            bp_tasks = ["block_pushing_p1", "block_pushing_p2"]
            for task in bp_tasks:
                # 筛选数据
                task_data = bp_results[(bp_results["Method"] == method) & 
                                     (bp_results["Steps"] == steps) & 
                                     (bp_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 计算最大性能和平均性能
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # 计算FLOPs和加速比
            flops = bp_results[(bp_results["Method"] == method) & 
                             (bp_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = bp_results[(bp_results["Method"] == method) & 
                             (bp_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 打印各方法的平均性能
    print("\n\033[1;34m----- Block-pushing任务方法性能比较 -----\033[0m")
    for method in sorted(bp_results["Method"].unique()):
        method_data = bp_results[bp_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}: 平均成功率 = {avg_success:.3f}, 平均加速比 = {avg_speedup:.2f}x")
# 处理Kitchen任务表格数据
if has_kitchen:
    print("\n\033[1;34m===== Kitchen任务 BAC消融实验表格数据 =====\033[0m")
    kitchen_results = results[results["TaskType"] == "kitchen"]
    
    # 生成LaTeX表格格式的结果
    for method in sorted(kitchen_results["Method"].unique()):
        for steps in sorted(kitchen_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # 处理p1到p4任务
            kitchen_tasks = ["kitchen_p1", "kitchen_p2", "kitchen_p3", "kitchen_p4"]
            for task in kitchen_tasks:
                # 筛选数据
                task_data = kitchen_results[(kitchen_results["Method"] == method) & 
                                         (kitchen_results["Steps"] == steps) & 
                                         (kitchen_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 计算最大性能和平均性能
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # 计算FLOPs和加速比
            flops = kitchen_results[(kitchen_results["Method"] == method) & 
                                 (kitchen_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = kitchen_results[(kitchen_results["Method"] == method) & 
                                  (kitchen_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 打印各方法的平均性能
    print("\n\033[1;34m----- Kitchen任务方法性能比较 -----\033[0m")
    for method in sorted(kitchen_results["Method"].unique()):
        method_data = kitchen_results[kitchen_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}: 平均成功率 = {avg_success:.3f}, 平均加速比 = {avg_speedup:.2f}x")
# 如果不同类型都有，打印总体性能比较
if (has_ph and has_mh) or (has_ph and has_bp) or (has_ph and has_kitchen) or (has_mh and has_bp) or (has_mh and has_kitchen) or (has_bp and has_kitchen):
    print("\n\033[1;34m===== 总体方法性能比较 =====\033[0m")
    for method in sorted(results["Method"].unique()):
        method_data = results[results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        
        # 按任务类型分组的统计信息
        task_type_stats = []
        if has_ph:
            ph_success = method_data[method_data["TaskType"] == "ph"]["SuccessRate"].mean()
            ph_speedup = method_data[method_data["TaskType"] == "ph"]["Speedup"].mean()
            task_type_stats.append(f"PH任务: 成功率 = {ph_success:.3f}, 加速比 = {ph_speedup:.2f}x")
        
        if has_mh:
            mh_success = method_data[method_data["TaskType"] == "mh"]["SuccessRate"].mean()
            mh_speedup = method_data[method_data["TaskType"] == "mh"]["Speedup"].mean()
            task_type_stats.append(f"MH任务: 成功率 = {mh_success:.3f}, 加速比 = {mh_speedup:.2f}x")
        
        if has_bp:
            bp_success = method_data[method_data["TaskType"] == "block_pushing"]["SuccessRate"].mean()
            bp_speedup = method_data[method_data["TaskType"] == "block_pushing"]["Speedup"].mean()
            task_type_stats.append(f"Block-pushing: 成功率 = {bp_success:.3f}, 加速比 = {bp_speedup:.2f}x")
        
        if has_kitchen:
            kitchen_success = method_data[method_data["TaskType"] == "kitchen"]["SuccessRate"].mean()
            kitchen_speedup = method_data[method_data["TaskType"] == "kitchen"]["Speedup"].mean()
            task_type_stats.append(f"Kitchen: 成功率 = {kitchen_success:.3f}, 加速比 = {kitchen_speedup:.2f}x")
        
        # 打印总体性能
        print(f"{method}: 总体成功率 = {avg_success:.3f}, 总体加速比 = {avg_speedup:.2f}x")
        # 打印每种任务类型的性能
        for stat in task_type_stats:
            print(f"  - {stat}")
EOF

# 创建任务汇总文件
print_title "创建任务汇总文件..."
for TASK_NAME in "${ACTIVE_TASKS[@]}"; do
    if [[ "$TASK_NAME" == *"_ph"* ]] || [[ "$TASK_NAME" == "pusht" ]]; then
        TASK_TYPE_NAME="ph"
    else
        TASK_TYPE_NAME="mh"
    fi

    TASK_SUMMARY_FILE="${BENCHMARK_RESULTS_DIR}/${TASK_NAME}_summary.csv"
    echo "Method,Steps,CacheMode,Metric,MaxCheckpointAvg,AvgCheckpointAvg,FLOPsAvg,SpeedupAvg" > "$TASK_SUMMARY_FILE"

    # 使用awk提取各方法和步数的平均值
    awk -F, -v task="$TASK_NAME" -v outfile="$TASK_SUMMARY_FILE" '
    NR == 1 {next} # 跳过标题行
    $3 == task {
        method=$1
        steps=$2
        checkpoint=$5
        rate=$6
        flops=$7
        speedup=$8
        
        # 跳过无效数据
        if (rate == "N/A" || rate == "-") next
        
        # 按方法、步数和检查点类型分组
        if (checkpoint == "max") {
            max_sum[method,steps] += rate
            max_count[method,steps]++
        } else if (checkpoint == "avg") {
            avg_sum[method,steps] += rate
            avg_count[method,steps]++
        }
        
        # 累计所有FLOPs和Speedup (不区分检查点类型)
        if (flops != "N/A" && flops != "-") {
            flops_sum[method,steps] += flops
            flops_count[method,steps]++
        }
        if (speedup != "N/A" && speedup != "-") {
            speedup_sum[method,steps] += speedup
            speedup_count[method,steps]++
        }
    }
    END {
        # 输出汇总结果
        for (key in max_sum) {
            split(key, parts, SUBSEP)
            method = parts[1]
            steps = parts[2]
            
            # 计算平均值
            max_avg = (max_count[key] > 0) ? max_sum[key]/max_count[key] : "N/A"
            avg_avg = (avg_count[key] > 0) ? avg_sum[key]/avg_count[key] : "N/A"
            flops_avg = (flops_count[key] > 0) ? flops_sum[key]/flops_count[key] : "N/A"
            speedup_avg = (speedup_count[key] > 0) ? speedup_sum[key]/speedup_count[key] : "N/A"
            
            # 添加固定的缓存模式和度量标准作为输出列
            printf "%s,%s,${cache_mode_var},${METRIC},%s,%s,%s,%s\n", method, steps, max_avg, avg_avg, flops_avg, speedup_avg >> outfile
        }
    }
    ' "$RESULTS_FILE"

    print_info "创建了任务 ${TASK_NAME} 的汇总文件: ${TASK_SUMMARY_FILE}"
done

# 创建最终汇总文件
FINAL_SUMMARY_FILE="${BENCHMARK_RESULTS_DIR}/final_summary.csv"
echo "Method,Steps,Task,TaskType,CacheMode,Metric,SuccessRate(MAX/AVG),FLOPs,Speedup" > "$FINAL_SUMMARY_FILE"

# 从各个任务汇总文件中收集数据
for TASK_NAME in "${ACTIVE_TASKS[@]}"; do
    if [[ "$TASK_NAME" == *"_ph"* ]] || [[ "$TASK_NAME" == "pusht" ]]; then
        TASK_TYPE_NAME="ph"
    else
        TASK_TYPE_NAME="mh"
    fi

    TASK_SUMMARY_FILE="${BENCHMARK_RESULTS_DIR}/${TASK_NAME}_summary.csv"

    # 如果任务汇总文件存在，添加到最终汇总
    if [ -f "$TASK_SUMMARY_FILE" ]; then
        tail -n +2 "$TASK_SUMMARY_FILE" | while IFS=, read -r method steps cache_mode metric max_avg avg_avg flops_avg speedup_avg; do
            # 格式化成功率为MAX/AVG形式
            if [ "$max_avg" = "N/A" ] && [ "$avg_avg" = "N/A" ]; then
                formatted_rate="-/-"
            elif [ "$max_avg" = "N/A" ]; then
                formatted_rate="-/${avg_avg}"
            elif [ "$avg_avg" = "N/A" ]; then
                formatted_rate="${max_avg}/-"
            else
                formatted_rate="${max_avg}/${avg_avg}"
            fi

            echo "${method},${steps},${TASK_NAME},${TASK_TYPE_NAME},${cache_mode},${metric},${formatted_rate},${flops_avg},${speedup_avg}" >> "$FINAL_SUMMARY_FILE"
        done
    fi
done

print_title "BAC消融实验完成！结果已保存到以下文件:"
echo "- 主结果文件: $RESULTS_FILE"
if [[ "$TASK_TYPE" == "ph" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "- PH任务结果: $PH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "mh" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "- MH任务结果: $MH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "block_pushing" ]]; then
    echo "- Block-pushing任务结果: $BP_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "kitchen" ]]; then
    echo "- Kitchen任务结果: $KITCHEN_RESULTS_FILE"
fi
echo "- 最终汇总文件: $FINAL_SUMMARY_FILE"

# 如果没有参数则显示使用说明
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi
