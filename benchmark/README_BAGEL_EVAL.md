# BAGEL在Smart Watch Benchmark上的评估

本文档说明如何使用Smart Watch Benchmark数据集评估BAGEL模型的性能，用于测试统一视觉语言模型在特定领域任务上的泛化能力。

## 📋 概述

**目标**：使用benchmark文件夹中的合成Smart Watch数据集测试BAGEL模型在视觉理解和生成任务上的表现。

**评估任务**：
- ✅ **VQA任务**：智能手表界面的视觉问答（时间识别、天气识别、位置识别、电量识别）
- ✅ **Caption任务**：智能手表界面的描述生成
- 🔄 **Generation任务**：根据描述生成智能手表界面图像（待完善）

## 🚀 快速开始

### 1. 准备环境

确保已经安装BAGEL的依赖：
```bash
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

### 2. 准备数据集

如果还没有生成Smart Watch测试数据集，可以使用提供的notebook生成：
```bash
# 进入benchmark目录
cd benchmark/Generalization_unified_VLM

# 使用generate_data_smart_watch.ipynb生成训练和测试数据
# 需要设置以下路径：
# - image_folder: 图像保存路径
# - data_file_path: JSON数据保存路径
# - num_of_samples: 样本数量
```

数据集格式示例：
```json
[
    {
        "task": "vqa",
        "image": "0.png",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is the current time?"},
            {"from": "gpt", "value": "12:30:45"}
        ]
    },
    {
        "task": "caption",
        "image": "0.png",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat are the key components of this image?"},
            {"from": "gpt", "value": "The image shows a smart watch, on which the current time is 12:30:45..."}
        ]
    }
]
```

### 3. 下载BAGEL模型

```bash
cd ../..  # 回到项目根目录

python -c "
from huggingface_hub import snapshot_download

save_dir = 'models/BAGEL-7B-MoT'
repo_id = 'ByteDance-Seed/BAGEL-7B-MoT'
cache_dir = save_dir + '/cache'

snapshot_download(
    cache_dir=cache_dir,
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=['*.json', '*.safetensors', '*.bin', '*.py', '*.md', '*.txt'],
)
"
```

### 4. 运行评估

**方法1：使用Shell脚本（推荐）**

编辑`benchmark/run_bagel_eval.sh`中的路径参数，然后运行：
```bash
bash benchmark/run_bagel_eval.sh
```

**方法2：直接使用Python命令**

```bash
python benchmark/eval_bagel_on_smartwatch.py \
    --model-path models/BAGEL-7B-MoT \
    --data-path benchmark/Generalization_unified_VLM/smart_watch_test.json \
    --image-folder benchmark/Generalization_unified_VLM/smart_watch_image_test \
    --output-dir benchmark/bagel_smartwatch_results \
    --device cuda:0 \
    --task-type vqa \
    --max-samples -1
```

## 📊 评估指标

### VQA任务指标

评估脚本会自动计算以下指标：

1. **时间识别准确率** (`time_acc`)
   - 评估模型识别手表时间（时:分:秒）的准确度
   - 使用误差归一化计算，容忍循环误差（如23:59和00:01）

2. **天气识别准确率** (`weather_acc`)
   - 评估模型识别天气图标（sunny/cloudy/raining）的准确度
   - 使用精确匹配

3. **位置识别准确率** (`position_acc`)
   - 评估模型识别UI元素位置（top-left/top-right/bottom-left/bottom-right）的准确度
   - 使用精确匹配

4. **电量识别准确率** (`battery_acc`)
   - 评估模型识别电池电量百分比的准确度
   - 使用误差归一化计算

5. **总体准确率** (`total_acc`)
   - 所有任务的平均准确率

### 结果输出

评估完成后，会在输出目录生成以下文件：

```
bagel_smartwatch_results/
├── vqa_results.json          # 详细的VQA推理结果
├── vqa_metrics.json          # VQA评估指标汇总
└── generation_results/       # 图像生成结果（待实现）
```

**vqa_results.json** 格式：
```json
[
    {
        "image": "0.png",
        "question": "What is the current time?",
        "ground_truth": "12:30:45",
        "prediction": "12:30:44",
        "task": "vqa"
    }
]
```

**vqa_metrics.json** 格式：
```json
{
    "time_acc": 0.95,
    "weather_acc": 0.98,
    "position_acc": 0.92,
    "battery_acc": 0.96,
    "total_acc": 0.95
}
```

## 🔧 参数说明

### 命令行参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--model-path` | str | 是 | BAGEL模型路径 |
| `--data-path` | str | 是 | 测试数据JSON文件路径 |
| `--image-folder` | str | 是 | 测试图像文件夹路径 |
| `--output-dir` | str | 否 | 结果输出目录（默认：`./bagel_smartwatch_results`） |
| `--device` | str | 否 | GPU设备（默认：`cuda:0`） |
| `--task-type` | str | 否 | 任务类型：`vqa`/`generation`/`all`（默认：`vqa`） |
| `--max-samples` | int | 否 | 最大评估样本数，-1表示全部（默认：-1） |

## 📈 预期结果

### BAGEL vs LLaVA-based方法

根据BAGEL的性能表现和Smart Watch Benchmark的特点，我们预期：

| 指标 | LLaVA-based (baseline) | BAGEL (expected) |
|------|------------------------|------------------|
| 时间识别 | ~0.90 | **>0.95** |
| 天气识别 | ~0.95 | **>0.98** |
| 位置识别 | ~0.88 | **>0.95** |
| 电量识别 | ~0.92 | **>0.96** |
| 总体准确率 | ~0.91 | **>0.96** |

**优势分析**：
- BAGEL的MoT架构在多模态理解任务上表现更优
- 更强的视觉编码器（SigLIP + VAE）提供更丰富的特征
- 更大规模的预训练数据提升泛化能力

## 🔍 问题排查

### 常见问题

1. **CUDA内存不足**
   - 解决方案：使用`--max-samples`限制样本数量，或使用量化版本的BAGEL

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确保已下载完整的模型文件

3. **图像加载错误**
   - 检查`--image-folder`路径是否正确
   - 确保图像文件存在且格式正确

## 🚧 待完善功能

### 图像生成任务评估

当前版本暂未实现完整的图像生成评估，因为需要：

1. **特殊的prompt格式**：BAGEL的图像生成需要特定的prompt模板
2. **解码流程**：需要实现latent到图像的解码
3. **FID计算**：需要生成足够数量的图像并计算FID分数

如需完整实现，可参考BAGEL的图像生成示例：
- `inferencer.py`中的图像生成代码
- `eval/gen/`目录下的生成评估脚本

## 📝 引用

如果使用此评估脚本，请引用：

```bibtex
@article{bagel2025,
  title={Emerging Properties in Unified Multimodal Pretraining},
  author={Deng, Chaorui and Zhu, Deyao and Li, Kunchang and others},
  journal={arXiv preprint arXiv:2505.14683},
  year={2025}
}
```

## 🤝 贡献

欢迎提交Issue和PR来改进评估脚本！

## 📧 联系

如有问题，请在GitHub Issue中提出或联系项目维护者。

