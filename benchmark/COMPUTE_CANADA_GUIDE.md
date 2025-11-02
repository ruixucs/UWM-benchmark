# 在Compute Canada上运行BAGEL评估

本指南说明如何在Compute Canada集群上提交SLURM作业来评估BAGEL模型。

## 📋 准备工作

### 1. 连接到Compute Canada

```bash
# 连接到Compute Canada集群（选择其中一个）
ssh username@graham.computecanada.ca   # Graham
ssh username@cedar.computecanada.ca    # Cedar
ssh username@beluga.computecanada.ca   # Beluga
ssh username@narval.computecanada.ca   # Narval
```

### 2. 设置环境

**创建conda环境**：
```bash
# 加载Python模块
module load python/3.10

# 创建虚拟环境
virtualenv --no-download $HOME/bagel_env

# 激活环境
source $HOME/bagel_env/bin/activate

# 或者使用conda（如果已安装）
conda create -n bagel python=3.10 -y
conda activate bagel
```

**安装依赖**：
```bash
# 进入项目目录
cd $HOME/projects/def-yourpi/yourname/UWM-benchmark

# 安装依赖
pip install --no-index -r requirements.txt

# 安装flash-attention（从预编译wheel）
pip install --no-index flash-attn
```

### 3. 下载模型和数据

**下载BAGEL模型**：
```bash
# 创建模型目录
mkdir -p $HOME/projects/def-yourpi/yourname/models

# 下载模型（在登录节点执行）
python << EOF
from huggingface_hub import snapshot_download

save_dir = "$HOME/projects/def-yourpi/yourname/models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(
    cache_dir=cache_dir,
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=['*.json', '*.safetensors', '*.bin', '*.py', '*.md', '*.txt'],
)
EOF
```

**生成测试数据**（如果还没有）：
```bash
# 使用notebook生成，或从其他地方传输
# 确保数据在：benchmark/Generalization_unified_VLM/smart_watch_test.json
# 图像在：benchmark/Generalization_unified_VLM/smart_watch_image_test/
```

### 4. 配置SLURM脚本

编辑 `benchmark/submit_bagel_eval_slurm.sh`，修改以下参数：

```bash
# 必须修改的参数
#SBATCH --account=def-yourpi              # 你的PI账号
#SBATCH --mail-user=your.email@example.com  # 你的邮箱

# 模型路径
MODEL_PATH="$HOME/projects/def-yourpi/yourname/models/BAGEL-7B-MoT"

# conda环境名称（如果使用conda）
conda activate bagel  # 改为你的环境名
```

**GPU选项**：
- `gpu:v100:1` - V100 GPU（推荐）
- `gpu:a100:1` - A100 GPU（更快，但可能需要等待）
- `gpu:p100:1` - P100 GPU（较旧，但通常可用）

**内存和时间**：
- 对于小数据集（<1000样本）：`--mem=16G --time=02:00:00`
- 对于完整数据集：`--mem=32G --time=04:00:00`

## 🚀 提交作业

### 提交评估作业

```bash
# 进入项目目录
cd $HOME/projects/def-yourpi/yourname/UWM-benchmark

# 提交作业
sbatch benchmark/submit_bagel_eval_slurm.sh
```

提交后会显示作业ID：
```
Submitted batch job 12345678
```

### 查看作业状态

```bash
# 查看你的所有作业
squeue -u $USER

# 查看特定作业的详细信息
scontrol show job 12345678

# 查看作业输出（实时）
tail -f logs/bagel_eval_12345678.out

# 查看错误日志
tail -f logs/bagel_eval_12345678.err
```

### 取消作业

```bash
# 取消特定作业
scancel 12345678

# 取消所有你的作业
scancel -u $USER
```

## 📊 查看结果

作业完成后，结果会保存在：
```
benchmark/bagel_smartwatch_results_<JOB_ID>/
├── vqa_results.json      # 详细推理结果
└── vqa_metrics.json      # 评估指标
```

**查看评估指标**：
```bash
# 查看VQA指标
cat benchmark/bagel_smartwatch_results_*/vqa_metrics.json | jq .

# 示例输出：
# {
#   "time_acc": 0.9542,
#   "weather_acc": 0.9834,
#   "position_acc": 0.9201,
#   "battery_acc": 0.9687,
#   "total_acc": 0.9566
# }
```

**下载结果到本地**：
```bash
# 在本地终端执行
scp -r username@graham.computecanada.ca:~/projects/def-yourpi/yourname/UWM-benchmark/benchmark/bagel_smartwatch_results_* ./
```

## ⚙️ 高级配置

### 1. 使用交互式GPU节点（调试用）

```bash
# 申请交互式GPU节点
salloc --account=def-yourpi --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00

# 等待分配后，加载模块并运行
module load python/3.10 cuda/12.1
source ~/bagel_env/bin/activate
cd $HOME/projects/def-yourpi/yourname/UWM-benchmark

# 直接运行评估
python benchmark/eval_bagel_on_smartwatch.py \
    --model-path $HOME/projects/def-yourpi/yourname/models/BAGEL-7B-MoT \
    --data-path benchmark/Generalization_unified_VLM/smart_watch_test.json \
    --image-folder benchmark/Generalization_unified_VLM/smart_watch_image_test \
    --output-dir benchmark/bagel_smartwatch_results_test \
    --device cuda:0 \
    --task-type vqa \
    --max-samples 100  # 先测试100个样本
```

### 2. 批量提交多个配置

创建一个批量提交脚本：
```bash
#!/bin/bash
# batch_submit.sh

for samples in 100 500 1000 -1; do
    JOB_ID=$(sbatch --export=MAX_SAMPLES=$samples benchmark/submit_bagel_eval_slurm.sh | awk '{print $4}')
    echo "Submitted job $JOB_ID with MAX_SAMPLES=$samples"
    sleep 1
done
```

### 3. 数组作业（并行评估）

如果要并行评估多个配置：
```bash
#SBATCH --array=0-3  # 4个并行任务

# 在脚本中使用$SLURM_ARRAY_TASK_ID区分配置
```

## 🔍 问题排查

### 常见错误

1. **模块加载失败**
   ```bash
   # 查看可用模块
   module spider python
   module spider cuda
   
   # 确保模块兼容
   module load python/3.10 cuda/12.1
   ```

2. **CUDA版本不匹配**
   ```bash
   # 检查CUDA版本
   nvidia-smi
   
   # 加载对应版本的CUDA模块
   module load cuda/11.8  # 或其他版本
   ```

3. **内存不足**
   ```bash
   # 增加内存请求
   #SBATCH --mem=64G
   
   # 或减少batch size/样本数
   --max-samples 500
   ```

4. **磁盘配额不足**
   ```bash
   # 检查配额
   diskusage_report
   
   # 清理缓存
   rm -rf ~/.cache/huggingface/hub/*
   ```

### 调试技巧

**测试GPU可用性**：
```bash
salloc --account=def-yourpi --gres=gpu:1 --time=0:30:00
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

**检查依赖**：
```bash
python -c "import torch, transformers, flash_attn; print('All imports successful')"
```

**查看详细错误**：
```bash
# 查看完整错误日志
less logs/bagel_eval_<JOB_ID>.err

# 搜索特定错误
grep -i "error\|exception\|failed" logs/bagel_eval_<JOB_ID>.err
```

## 📈 性能优化

### 1. 使用快速本地存储

```bash
# 使用$SLURM_TMPDIR（节点本地SSD）
cp -r benchmark/Generalization_unified_VLM/smart_watch_image_test $SLURM_TMPDIR/
IMAGE_FOLDER=$SLURM_TMPDIR/smart_watch_image_test
```

### 2. 多GPU评估

修改脚本支持多GPU：
```bash
#SBATCH --gres=gpu:2

# 使用DataParallel或设置多个进程
```

### 3. 减少I/O开销

```bash
# 预加载数据到内存
# 使用更高效的数据格式（HDF5/LMDB）
```

## 📧 获取帮助

- **Compute Canada文档**: https://docs.computecanada.ca/
- **技术支持**: support@computecanada.ca
- **SLURM文档**: https://slurm.schedmd.com/

## 📝 资源估算

| 数据集大小 | GPU | 内存 | 时间 | 预估成本（核心小时）|
|-----------|-----|------|------|---------------------|
| 100样本    | 1xV100 | 16G | 0.5h | ~2 core-hours |
| 500样本    | 1xV100 | 24G | 1.5h | ~6 core-hours |
| 1000样本   | 1xV100 | 32G | 3h   | ~12 core-hours |
| 完整数据集  | 1xA100 | 32G | 4-6h | ~20-30 core-hours |

💡 **提示**：先用小数据集测试（`--max-samples 100`），确认脚本正常运行后再提交完整评估。

