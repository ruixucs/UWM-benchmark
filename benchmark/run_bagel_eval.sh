#!/bin/bash

# BAGEL在Smart Watch Benchmark上的评估脚本

# 配置参数
MODEL_PATH="models/BAGEL-7B-MoT"  # BAGEL模型路径
DATA_PATH="benchmark/Generalization_unified_VLM/smart_watch_test.json"  # 测试数据路径
IMAGE_FOLDER="benchmark/Generalization_unified_VLM/smart_watch_image_test"  # 测试图像文件夹
OUTPUT_DIR="benchmark/bagel_smartwatch_results"  # 输出目录
DEVICE="cuda:0"  # GPU设备
TASK_TYPE="vqa"  # 任务类型: vqa, generation, 或 all

# 可选：限制评估样本数量（用于快速测试）
MAX_SAMPLES=-1  # -1表示评估所有样本

echo "========================================"
echo "BAGEL Smart Watch Benchmark Evaluation"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Task Type: $TASK_TYPE"
echo "Device: $DEVICE"
echo "========================================"

# 运行评估
python benchmark/eval_bagel_on_smartwatch.py \
    --model-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --image-folder $IMAGE_FOLDER \
    --output-dir $OUTPUT_DIR \
    --device $DEVICE \
    --task-type $TASK_TYPE \
    --max-samples $MAX_SAMPLES

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

