# Llama-3.2-3B LoRA 微调项目

基于 **Llama-3.2-3B-Instruct** 模型的 LoRA (Low-Rank Adaptation) 微调项目，用于法律领域问答任务。

## 📋 项目简介

本项目使用 PEFT (Parameter-Efficient Fine-Tuning) 库中的 LoRA 技术对 Llama-3.2-3B 模型进行高效微调。通过在模型的注意力层中添加低秩矩阵，只需训练约 1.5% 的参数即可实现模型适配。

### 核心特性

- ✅ **参数高效**: 只训练 1.5% 的参数（~48M / 3.2B）
- ✅ **显存友好**: 支持 macOS MPS / CUDA GPU / CPU 训练
- ✅ **自动适配**: 自动检测设备并优化配置
- ✅ **快速训练**: 1000条数据约 30-45 分钟（M2 Max）
- ✅ **完整工具链**: 训练 → 推理 → 对比评估

### 数据集

- **文件**: `datasets/19503488-349b-4321-941d-7875fca0737b.csv`
- **样本数**: 1,000 条法律问答对
- **格式**: instruction (问题) + output (回答)
- **领域**: 中国法律（合同法、劳动法、刑法等）

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目（如果从 Git 获取）
git clone <your-repo>
cd lora

# 安装依赖
pip install -r requirements.txt
```

**系统要求**:
- Python 3.8+
- macOS (Apple Silicon) / Linux with CUDA / Windows with CUDA
- 内存: 8GB+ (macOS) / 显存: 8GB+ (CUDA)

### 2. 下载模型

```bash
# 创建下载脚本
cat > download_model.py << 'EOF'
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', 
                               cache_dir='./models', 
                               revision='master')
print(f"模型已下载到: {model_path}")
EOF

# 运行下载
python download_model.py
```

模型会下载到 `./models/LLM-Research/Llama-3.2-3B-Instruct/` (~6GB)

### 3. 开始训练

```bash
# 使用便捷脚本（推荐）
./train_csv.sh

# 或直接运行 Python
python train.py --dataset_name ./datasets/19503488-349b-4321-941d-7875fca0737b.csv
```

训练输出会保存到 `./output/llama-lora-YYYYMMDD_HHMMSS/`

### 4. 模型推理

```bash
# 单次推理
python inference.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --prompt "劳动合同纠纷如何处理？"

# 交互式对话
python inference.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --interactive
```

### 5. 效果对比

```bash
# 对比微调前后的回答
python compare_models.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --test_file test_questions.txt
```

## 📊 训练配置

### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `./models/LLM-Research/Llama-3.2-3B-Instruct` | 模型路径 |
| `dataset_name` | `./datasets/xxx.csv` | 数据集路径 |
| `num_epochs` | 3 | 训练轮数 |
| `batch_size` | 2 | 批次大小 |
| `gradient_accumulation_steps` | 8 | 梯度累积（有效批次=16） |
| `learning_rate` | 1e-4 | 学习率 |
| `max_length` | 512 | 最大序列长度 |
| `lora_r` | 32 | LoRA 秩 |
| `lora_alpha` | 64 | LoRA alpha |
| `lora_dropout` | 0.1 | LoRA dropout |

### LoRA 参数详解

- **lora_r**: 控制低秩矩阵的秩，越大模型容量越大（推荐: 16-64）
- **lora_alpha**: 缩放因子，通常设为 r 的 2 倍
- **lora_dropout**: 防止过拟合，法律领域推荐 0.1
- **target_modules**: 应用 LoRA 的层（已优化为全部注意力层和FFN层）

### 自定义训练

```bash
python train.py \
  --dataset_name ./datasets/19503488-349b-4321-941d-7875fca0737b.csv \
  --num_epochs 5 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --max_length 384 \
  --lora_r 16 \
  --lora_alpha 32
```

## ⏱️ 训练时间预估

### 1,000 样本，3 epochs

| 设备 | 配置 | 预计时间 |
|------|------|---------|
| **CUDA GPU** | | |
| RTX 3090 (24GB) | batch=2, 4-bit量化 | 20-30 分钟 |
| RTX 4090 (24GB) | batch=4, 4-bit量化 | 15-20 分钟 |
| A100 (40GB) | batch=4, 4-bit量化 | 10-15 分钟 |
| **Apple Silicon (macOS)** | | |
| M1 (8GB) | batch=1, fp32 | 2-3 小时 |
| M1 Pro (16GB) | batch=1, fp32 | 1-2 小时 |
| M2 Max (32GB) | batch=2, fp32 | 30-45 分钟 |
| M3 Max (64GB) | batch=4, fp32 | 20-30 分钟 |

## 💻 设备适配说明

### macOS (Apple Silicon)

项目自动适配 macOS：
- ✅ 自动检测并使用 **MPS** (Metal Performance Shaders) 加速
- ✅ 自动禁用 4-bit 量化（macOS 不支持）
- ✅ 使用 float32 精度（MPS 最佳兼容）
- ✅ 优化器自动切换为 `adamw_torch`

**注意**: macOS 上训练速度比 CUDA 慢约 1.5-2 倍，但仍然可用。

### CUDA GPU

- ✅ 支持 4-bit 量化（需要安装 `bitsandbytes`）
- ✅ 使用 bfloat16 混合精度训练
- ✅ 优化器：`paged_adamw_8bit`（量化）或 `adamw_torch`

### CPU

- ⚠️ 不推荐（训练非常慢）
- 建议使用云 GPU 服务（Google Colab、AWS 等）

## 📁 项目结构

```
lora/
├── train.py                    # 训练脚本
├── inference.py                # 推理脚本
├── compare_models.py           # 模型对比工具
├── train_csv.sh               # 训练启动脚本
├── config.yaml                # 配置文件
├── requirements.txt           # Python依赖
├── test_questions.txt         # 测试问题
├── README.md                  # 项目说明（本文件）
├── CSV_TRAINING_GUIDE.md      # CSV训练详细指南
├── .gitignore                 # Git忽略文件
│
├── datasets/                  # 数据集目录
│   └── 19503488-349b-4321-941d-7875fca0737b.csv  # 训练数据
│
├── models/                    # 模型目录
│   └── LLM-Research/
│       └── Llama-3.2-3B-Instruct/  # 下载的基础模型
│
└── output/                    # 训练输出（.gitignore已忽略）
    └── llama-lora-YYYYMMDD_HHMMSS/
        ├── final_model/       # 最终LoRA权重
        └── logs/             # TensorBoard日志
```

## 🎯 使用场景

### 1. 训练自定义模型

```bash
# 准备你的CSV数据（格式: instruction, output, input）
# 参考 datasets/19503488-349b-4321-941d-7875fca0737b.csv

python train.py \
  --dataset_name ./datasets/your_data.csv \
  --num_epochs 3
```

### 2. 调整训练参数

```bash
# 显存不足时
python train.py \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 256

# 提高训练质量
python train.py \
  --num_epochs 5 \
  --learning_rate 5e-5 \
  --lora_r 64 \
  --lora_alpha 128
```

### 3. 批量推理

```python
# batch_inference.py
from inference import load_model, generate_response

model, tokenizer = load_model(base_model, lora_weights)

questions = [
    "什么是合同法？",
    "劳动合同纠纷如何处理？",
    # ...
]

for q in questions:
    response = generate_response(model, tokenizer, q, args)
    print(f"Q: {q}\nA: {response}\n")
```

### 4. 模型评估

```bash
# 使用测试集评估
python compare_models.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --test_file test_questions.txt \
  > evaluation_results.txt

# 分析结果
grep "微调回答长度" evaluation_results.txt | awk '{print $2}'
```

## 🔧 常见问题

### Q1: 训练时显存不足 (OOM)

**解决方案**:
```bash
# 减小批次大小和序列长度
python train.py \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 256
```

### Q2: macOS 上训练很慢

这是正常的，macOS MPS 比 CUDA 慢。**建议**:
- 使用更小的数据集测试
- 减少 epochs
- 考虑使用云 GPU（Google Colab 免费提供 T4）

### Q3: 如何恢复中断的训练

Trainer 会自动保存 checkpoint，在 output 目录查找：
```bash
ls ./output/llama-lora-xxx/checkpoint-*/
```

然后修改 `train.py` 的 `resume_from_checkpoint` 参数。

### Q4: 微调后效果不明显

**可能原因**:
1. 训练数据太少 → 增加数据量
2. 训练轮数不够 → 增加 epochs
3. LoRA 秩太小 → 增大 lora_r
4. 学习率太小 → 适当提高

**验证方法**:
```bash
python compare_models.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --prompt "训练集中的问题"
```

### Q5: 如何评估模型质量

使用对比工具：
```bash
python compare_models.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --test_file test_questions.txt
```

关注：
- ✅ 是否引用法条
- ✅ 回答是否更详细
- ✅ 逻辑是否更清晰
- ✅ 专业术语使用是否准确

## 📈 监控训练

### 使用 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/llama-lora-xxx/logs

# 在浏览器打开
# http://localhost:6006
```

可以查看：
- 训练损失曲线
- 学习率变化
- 梯度统计信息

### 实时日志

训练过程会实时显示：
```
Epoch 1/3: 100%|████████| 63/63 [05:23<00:00, 5.16s/it]
{'loss': 1.234, 'learning_rate': 0.0001, 'epoch': 1.0}
```

## 🌟 进阶使用

### 1. 使用配置文件

编辑 `config.yaml` 然后：
```python
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 使用配置训练
# 修改 train.py 以支持配置文件
```

### 2. 多数据集训练

```python
import pandas as pd

# 合并多个数据集
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
combined = pd.concat([df1, df2], ignore_index=True)
combined.to_csv('combined_dataset.csv', index=False)

# 训练
python train.py --dataset_name combined_dataset.csv
```

### 3. 超参数搜索

```bash
# 测试不同的学习率
for lr in 5e-5 1e-4 2e-4; do
  python train.py --learning_rate $lr --output_dir ./output/lr_$lr
done

# 对比结果
python compare_models.py --lora_weights ./output/lr_5e-5/final_model
python compare_models.py --lora_weights ./output/lr_1e-4/final_model
python compare_models.py --lora_weights ./output/lr_2e-4/final_model
```

## 📚 相关资源

- [Llama 3.2 模型](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [ModelScope 平台](https://modelscope.cn/)

## 📄 许可证

本项目代码遵循 MIT 许可证。

**注意**: Llama 3.2 模型有自己的使用条款，请遵守 [Meta Llama 3.2 社区许可协议](https://www.llama.com/llama3_2/license/)。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

改进建议：
- 支持更多数据格式
- 添加更多评估指标
- 优化训练速度
- 支持多GPU训练

## 📧 联系方式

如有问题，请：
1. 查看 `CSV_TRAINING_GUIDE.md` 获取更多细节
2. 提交 GitHub Issue
3. 参考常见问题部分

---

## 快速命令速查

```bash
# 训练
./train_csv.sh

# 推理
python inference.py --lora_weights ./output/xxx/final_model --interactive

# 对比
python compare_models.py --lora_weights ./output/xxx/final_model --test_file test_questions.txt

# 监控
tensorboard --logdir ./output/xxx/logs

# 清理
rm -rf ./output/*
```

祝你微调愉快！🚀
