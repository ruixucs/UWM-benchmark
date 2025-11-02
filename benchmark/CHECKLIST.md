# ✅ Compute Canada提交前检查清单

在提交SLURM作业前，请按照此清单逐项检查：

## 📝 必须完成的配置

### 1. 编辑SLURM脚本参数
打开 `benchmark/submit_bagel_eval_slurm.sh`，修改：

- [ ] `#SBATCH --account=def-yourpi` → 改为你的实际PI账号
- [ ] `#SBATCH --mail-user=your.email@example.com` → 改为你的邮箱
- [ ] `MODEL_PATH` → 改为你的模型实际路径
- [ ] `conda activate bagel` → 改为你的环境名（或改用virtualenv）

### 2. 确认环境准备
- [ ] 已创建并激活Python环境（conda或virtualenv）
- [ ] 已安装所有依赖：`pip install -r requirements.txt`
- [ ] 已安装flash-attention
- [ ] 测试过Python和CUDA可用性

### 3. 确认数据和模型存在
- [ ] BAGEL模型已下载到指定路径
- [ ] 测试数据JSON文件存在
- [ ] 测试图像文件夹存在且包含图像
- [ ] 路径使用绝对路径或相对于项目根目录的路径

### 4. 创建必要的目录
```bash
mkdir -p logs
```

## 🚀 提交流程

### 快速开始（5分钟测试）

1. **先测试少量样本**（推荐！）
   ```bash
   # 编辑submit_bagel_eval_slurm.sh，设置：
   MAX_SAMPLES=100
   #SBATCH --time=00:30:00
   ```

2. **提交测试作业**
   ```bash
   sbatch benchmark/submit_bagel_eval_slurm.sh
   ```

3. **监控作业**
   ```bash
   # 记下作业ID，例如12345678
   watch -n 10 squeue -u $USER
   
   # 查看实时日志
   tail -f logs/bagel_eval_12345678.out
   ```

4. **检查结果**
   - 如果测试成功，修改`MAX_SAMPLES=-1`提交完整评估
   - 如果失败，查看错误日志：`logs/bagel_eval_<JOB_ID>.err`

### 完整评估

测试成功后：
```bash
# 1. 修改submit_bagel_eval_slurm.sh
MAX_SAMPLES=-1
#SBATCH --time=04:00:00

# 2. 提交完整作业
sbatch benchmark/submit_bagel_eval_slurm.sh
```

## 🔧 常见配置选项

### GPU类型选择
```bash
# V100 (推荐，平衡性能和可用性)
#SBATCH --gres=gpu:v100:1

# A100 (更快，但可能需要等待)
#SBATCH --gres=gpu:a100:1

# P100 (较旧，但通常立即可用)
#SBATCH --gres=gpu:p100:1
```

### 资源配置
```bash
# 小数据集 (<500样本)
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# 中等数据集 (500-2000样本)
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# 大数据集 (>2000样本)
#SBATCH --mem=32G
#SBATCH --time=04:00:00
```

## 📊 提交后命令速查

```bash
# 查看作业队列
squeue -u $USER

# 查看作业详情
scontrol show job <JOB_ID>

# 实时查看输出
tail -f logs/bagel_eval_<JOB_ID>.out

# 查看错误日志
tail -f logs/bagel_eval_<JOB_ID>.err

# 取消作业
scancel <JOB_ID>

# 查看资源使用
seff <JOB_ID>
```

## ⚠️ 常见问题快速解决

### 问题1：作业一直在PENDING状态
**原因**：等待资源分配
**解决**：
- 查看原因：`squeue -u $USER -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %.20R"`
- 考虑换一个GPU类型
- 或减少资源请求（内存、时间）

### 问题2：模块加载失败
```bash
# 查看可用模块版本
module spider python
module spider cuda

# 加载正确版本
module load python/3.10 cuda/12.1
```

### 问题3：找不到Python包
```bash
# 确认环境已激活
which python
pip list | grep torch

# 重新安装
pip install --upgrade torch transformers
```

### 问题4：CUDA内存不足
```bash
# 在submit_bagel_eval_slurm.sh中添加：
MAX_SAMPLES=500  # 减少样本数

# 或请求更多GPU
#SBATCH --gres=gpu:a100:1  # A100有40GB显存
```

## 📁 文件结构检查

确保你的项目结构如下：
```
UWM-benchmark/
├── benchmark/
│   ├── submit_bagel_eval_slurm.sh     ✓ SLURM提交脚本
│   ├── eval_bagel_on_smartwatch.py    ✓ 评估脚本
│   ├── Generalization_unified_VLM/
│   │   ├── smart_watch_test.json      ✓ 测试数据
│   │   └── smart_watch_image_test/    ✓ 测试图像
│   └── logs/                          ✓ 日志目录
├── inferencer.py                      ✓ BAGEL推理器
├── data/                              ✓ 数据加载模块
└── models/
    └── BAGEL-7B-MoT/                  ✓ 模型文件
```

## 🎯 提交前最后确认

运行这个一键检查脚本：
```bash
#!/bin/bash
echo "=== Checking configuration ==="

# 检查SLURM脚本
if grep -q "def-yourpi" benchmark/submit_bagel_eval_slurm.sh; then
    echo "❌ 请修改 --account 参数"
else
    echo "✓ Account configured"
fi

# 检查模型路径
MODEL_PATH=$(grep "^MODEL_PATH=" benchmark/submit_bagel_eval_slurm.sh | cut -d'"' -f2)
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model exists: $MODEL_PATH"
else
    echo "❌ Model not found: $MODEL_PATH"
fi

# 检查数据
if [ -f "benchmark/Generalization_unified_VLM/smart_watch_test.json" ]; then
    echo "✓ Test data exists"
else
    echo "❌ Test data not found"
fi

# 检查Python环境
if command -v python &> /dev/null; then
    echo "✓ Python available: $(python --version)"
else
    echo "❌ Python not found"
fi

echo "=== Check complete ==="
```

## ✨ 快速提交命令

如果所有检查都通过，执行：
```bash
cd ~/projects/def-yourpi/yourname/UWM-benchmark
sbatch benchmark/submit_bagel_eval_slurm.sh
```

祝评估顺利！🚀

