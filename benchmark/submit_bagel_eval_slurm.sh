#!/bin/bash
#SBATCH --job-name=bagel_smartwatch_eval    # 作业名称
#SBATCH --account=def-yourpi                # 替换为你的PI账号
#SBATCH --time=04:00:00                     # 最大运行时间（4小时）
#SBATCH --nodes=1                           # 节点数
#SBATCH --ntasks=1                          # 任务数
#SBATCH --cpus-per-task=4                   # 每个任务的CPU核心数
#SBATCH --mem=32G                           # 内存
#SBATCH --gres=gpu:v100:1                   # GPU资源（1个V100），可选：a100, p100
#SBATCH --output=logs/bagel_eval_%j.out     # 标准输出文件
#SBATCH --error=logs/bagel_eval_%j.err      # 标准错误文件
#SBATCH --mail-type=ALL                     # 邮件通知类型
#SBATCH --mail-user=your.email@example.com  # 替换为你的邮箱

# ============================================================================
# BAGEL在Smart Watch Benchmark上的评估 - Compute Canada SLURM作业脚本
# ============================================================================

echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# 创建日志目录
mkdir -p logs

# 加载必要的模块
module load python/3.10
module load cuda/12.1
module load gcc/9.3.0

echo "Loaded modules:"
module list

# 激活conda环境（如果使用conda）
# 替换为你的conda环境名称
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bagel

echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 进入项目目录
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# ============================================================================
# 配置路径参数
# ============================================================================

# BAGEL模型路径（需要提前下载到Compute Canada存储）
MODEL_PATH="$HOME/projects/def-yourpi/yourname/models/BAGEL-7B-MoT"

# Smart Watch数据集路径
DATA_PATH="benchmark/Generalization_unified_VLM/smart_watch_test.json"
IMAGE_FOLDER="benchmark/Generalization_unified_VLM/smart_watch_image_test"

# 输出目录
OUTPUT_DIR="benchmark/bagel_smartwatch_results_${SLURM_JOB_ID}"

# GPU设备
DEVICE="cuda:0"

# 任务类型：vqa, generation, 或 all
TASK_TYPE="vqa"

# 可选：限制评估样本数量（-1表示全部）
MAX_SAMPLES=-1

echo ""
echo "========================================"
echo "Configuration"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Image Folder: $IMAGE_FOLDER"
echo "Output Dir: $OUTPUT_DIR"
echo "Task Type: $TASK_TYPE"
echo "Device: $DEVICE"
echo "Max Samples: $MAX_SAMPLES"
echo "========================================"
echo ""

# ============================================================================
# 检查模型和数据是否存在
# ============================================================================

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    echo "Please download BAGEL model first using:"
    echo "  python -c \"from huggingface_hub import snapshot_download; ..."
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file does not exist: $DATA_PATH"
    echo "Please generate test data first"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "ERROR: Image folder does not exist: $IMAGE_FOLDER"
    echo "Please generate test images first"
    exit 1
fi

# ============================================================================
# 运行评估
# ============================================================================

echo "Starting evaluation..."
echo ""

python benchmark/eval_bagel_on_smartwatch.py \
    --model-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --image-folder $IMAGE_FOLDER \
    --output-dir $OUTPUT_DIR \
    --device $DEVICE \
    --task-type $TASK_TYPE \
    --max-samples $MAX_SAMPLES

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # 显示结果摘要
    if [ -f "$OUTPUT_DIR/vqa_metrics.json" ]; then
        echo ""
        echo "VQA Metrics:"
        cat $OUTPUT_DIR/vqa_metrics.json
    fi
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "========================================"

# 资源使用统计
echo ""
echo "Job finished at: $(date)"
echo "========================================"
echo "Resource Usage:"
seff $SLURM_JOB_ID

exit $EXIT_CODE

