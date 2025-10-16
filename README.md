# macOS 训练指南

## ⚠️ macOS 特别说明

你正在 macOS 上运行此项目。由于 macOS 不支持 CUDA，训练配置已自动调整。

## 🔄 自动调整内容

### 1. 禁用 4-bit 量化
- ❌ **bitsandbytes** 库需要 CUDA，macOS 不可用
- ✅ 改用 **全精度 (float32)** 或 **半精度 (float16)**
- ✅ 显存占用会增加，但在 Apple Silicon 上仍可运行

### 2. 使用 MPS 加速
- ✅ **MPS (Metal Performance Shaders)** - Apple 的 GPU 加速框架
- ✅ 适用于 M1/M2/M3 系列芯片
- ⚠️ 性能比 CUDA 慢，但比 CPU 快很多

### 3. 优化器调整
- ❌ `paged_adamw_8bit` (需要 bitsandbytes)
- ✅ `adamw_torch` (PyTorch 原生优化器)

## 💻 硬件要求

### Apple Silicon (M 系列芯片) - 推荐

| 芯片 | 统一内存 | 可训练模型 | 预估速度 |
|------|---------|----------|---------|
| M1 | 8 GB | 1B 模型 | 慢 |
| M1 Pro/Max | 16-32 GB | 3B 模型 | 中等 |
| M2 | 8-24 GB | 1-3B 模型 | 中等 |
| M2 Pro/Max | 16-96 GB | 3-7B 模型 | 快 |
| M3 | 8-24 GB | 1-3B 模型 | 快 |
| M3 Pro/Max | 18-128 GB | 3-13B 模型 | 很快 |

### Intel Mac - 不推荐

- ⚠️ 仅支持 CPU 训练
- ⚠️ 速度非常慢（比 M1 慢 10-50 倍）
- 💡 建议使用云 GPU（如 Google Colab, AWS）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: `bitsandbytes` 已被注释掉，无需安装。

### 2. 开始训练

```bash
# 使用默认配置（会自动适配 macOS）
python train.py

# 或使用脚本
./run_train.sh
```

训练脚本会自动：
- ✅ 检测 MPS 可用性
- ✅ 禁用 4-bit 量化
- ✅ 使用 float32 精度
- ✅ 选择合适的优化器

## ⚙️ macOS 优化配置

### 推荐参数（Llama-3.2-3B + M1 Pro/Max）

```bash
python train.py \
  --batch_size 1 \              # 减小批次节省内存
  --gradient_accumulation_steps 16 \  # 增加累积保持效果
  --max_length 384 \            # 减小序列长度
  --num_epochs 2 \              # 减少轮数加快训练
  --lora_r 16 \                 # 减小 LoRA 秩
  --lora_alpha 32
```

### 如果内存不足

```bash
python train.py \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 256 \
  --lora_r 8 \
  --lora_alpha 16
```

## ⏱️ 训练时间预估

### Llama-3.2-3B 模型（21,476 样本，3 epochs）

| 设备 | 配置 | 预计时间 |
|------|------|---------|
| M1 (8GB) | batch=1, len=256 | 48-72 小时 |
| M1 Pro (16GB) | batch=1, len=384 | 24-36 小时 |
| M2 Max (32GB) | batch=2, len=512 | 12-18 小时 |
| M3 Max (64GB) | batch=4, len=512 | 8-12 小时 |

**对比 CUDA (RTX 3090)**:
- RTX 3090 (4-bit): ~9-12 小时
- M2 Max (fp32): ~12-18 小时 (约 1.5x 慢)

## 🔍 性能监控

### 检查 MPS 使用情况

```bash
# 终端监控
sudo powermetrics --samplers gpu_power -i1000 -n1

# 活动监控器
打开"活动监控器" → "能源" 标签页 → 查看 GPU 活动
```

### TensorBoard

```bash
tensorboard --logdir ./output/llama-lora-*/logs
```

## 💡 macOS 训练技巧

### 1. 优化内存使用

```python
# 在训练开始前添加
import torch
torch.mps.empty_cache()  # 清空 MPS 缓存
```

### 2. 防止 Mac 休眠

```bash
# 训练期间保持唤醒（新终端窗口）
caffeinate -i python train.py
```

### 3. 使用数据子集快速测试

```python
# 修改 train.py 中的数据集加载
dataset = load_dataset("Skepsun/lawyer_llama_data", split="train[:1000]")
# 只使用前 1000 个样本测试
```

## 🐛 常见问题

### Q1: 出现 "MPS backend out of memory" 错误

**解决方案**:
```bash
python train.py \
  --batch_size 1 \
  --max_length 256 \
  --gradient_accumulation_steps 32
```

### Q2: 训练速度很慢

**原因**:
- MPS 确实比 CUDA 慢（正常现象）
- Intel Mac 只能用 CPU（非常慢）

**建议**:
1. 减小数据集：使用子集快速测试
2. 减少 epochs：从 3 降到 1-2
3. 考虑使用云 GPU：
   - Google Colab (免费 T4 GPU)
   - AWS SageMaker
   - Paperspace Gradient

### Q3: 如何查看是否使用了 MPS？

运行训练时，应该看到：
```
✓ 检测到 Apple Silicon (MPS)
⚠️  macOS 不支持 4-bit 量化，已自动禁用
✓ 使用 float32 精度（MPS 最佳兼容）
✓ 模型已加载到 mps
```

### Q4: 可以在 macOS 上使用 CUDA 吗？

❌ **不可以**。macOS 不支持 NVIDIA CUDA。
- Apple Silicon 使用 **MPS**
- Intel Mac 只能用 **CPU**

## 🌐 云训练替代方案

如果 macOS 训练太慢，考虑使用云服务：

### Google Colab (推荐新手)
- ✅ 免费 T4 GPU (15GB)
- ✅ 无需配置
- ⚠️ 会话有时间限制

### Kaggle Notebooks
- ✅ 免费 P100 GPU (16GB)
- ✅ 每周 30 小时

### AWS SageMaker
- 💰 按需付费
- ✅ 强大的 A100/V100 GPU
- ✅ 适合大规模训练

### Paperspace Gradient
- 💰 按小时计费（~$0.5-2/小时）
- ✅ 简单易用
- ✅ 支持 Jupyter Notebook

## 📝 最佳实践

### macOS 上的推荐工作流

1. **本地开发调试**:
   ```bash
   # 使用小数据集快速测试代码
   python train.py --num_epochs 1 \
     --dataset_name "Skepsun/lawyer_llama_data" \
     # 限制样本数进行测试
   ```

2. **云端完整训练**:
   - 将代码上传到 Google Colab
   - 使用免费 GPU 完成完整训练
   - 下载训练好的模型

3. **本地推理测试**:
   ```bash
   # 下载训练好的模型后，在本地测试
   python inference.py --lora_weights ./output/.../final_model --interactive
   ```

## 🆚 macOS vs CUDA 对比

| 特性 | macOS (MPS) | CUDA GPU |
|------|-------------|----------|
| 4-bit 量化 | ❌ | ✅ |
| 训练速度 | 中等 | 快 |
| 显存效率 | 一般 | 优秀 |
| 功耗 | 低 | 高 |
| 成本 | 硬件贵 | 硬件便宜 |
| 便携性 | ✅ 笔记本 | ❌ 台式机 |
| 噪音 | 静音 | 可能大 |

## 📚 相关资源

- [Apple MPS 文档](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS 后端](https://pytorch.org/docs/stable/notes/mps.html)
- [PEFT 库文档](https://huggingface.co/docs/peft)

---

**总结**: macOS 可以训练 LoRA 模型，但速度较慢。适合开发调试，正式训练建议使用云 GPU。

如有问题，请参考 README.md 或提交 Issue。

