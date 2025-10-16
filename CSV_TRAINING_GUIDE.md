# CSV 数据集训练指南

## 📋 CSV 数据集说明

### 数据集信息
- **文件**: `datasets/19503488-349b-4321-941d-7875fca0737b.csv`
- **样本数**: 1,000 条
- **格式**: CSV (逗号分隔)

### 数据格式

CSV 文件包含 3 列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `input` | 额外的输入内容 | 通常为空字符串 "" |
| `instruction` | 问题/指令 | "分析挪用特定款物罪的法律后果..." |
| `output` | 期望的回答 | "挪用特定款物罪是指..." |

### 数据示例

```csv
input,instruction,output
"",分析挪用特定款物罪的法律后果...,挪用特定款物罪是指...
"",分析在运送他人偷越国（边）境...,根据文本内容，如果在运送...
```

## 🚀 快速开始

### 方式 1: 使用便捷脚本（推荐）

```bash
./train_csv.sh
```

脚本会自动：
- ✅ 检查 CSV 文件是否存在
- ✅ 显示样本数量
- ✅ 使用优化的参数开始训练

### 方式 2: 直接运行 Python（自定义参数）

```bash
python train.py \
  --dataset_name ./datasets/19503488-349b-4321-941d-7875fca0737b.csv \
  --dataset_type csv
```

### 方式 3: 自动检测（最简单）

```bash
# 只要文件名以 .csv 结尾，会自动检测为 CSV 格式
python train.py \
  --dataset_name ./datasets/19503488-349b-4321-941d-7875fca0737b.csv
```

## ⚙️ 训练参数

### 默认参数（适合 3B 模型）

```bash
python train.py \
  --dataset_name ./datasets/19503488-349b-4321-941d-7875fca0737b.csv \
  --model_name ./models/LLM-Research/Llama-3.2-3B-Instruct \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_length 512 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1
```

### macOS 优化参数（内存受限）

```bash
python train.py \
  --dataset_name ./datasets/19503488-349b-4321-941d-7875fca0737b.csv \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 384 \
  --lora_r 16 \
  --lora_alpha 32 \
  --num_epochs 2
```

## 📊 数据集对比

| 特性 | CSV 数据集 | 原 HuggingFace 数据集 |
|------|-----------|---------------------|
| 样本数 | 1,000 | 21,476 |
| 格式 | CSV 文件 | Arrow 缓存 |
| 加载速度 | 快 | 很快 |
| 训练时间 | 约 30-45 分钟 (M2 Max) | 约 12-18 小时 (M2 Max) |
| 磁盘占用 | ~1 MB | ~46 MB |

**优势**:
- ✅ **样本数较少**，训练更快（约为原数据集的 1/20）
- ✅ **适合快速实验**和调试
- ✅ **文件小**，易于编辑和查看
- ✅ **格式简单**，便于自定义数据

## ⏱️ 训练时间预估

### CSV 数据集 (1,000 样本，3 epochs)

| 设备 | 配置 | 预计时间 |
|------|------|---------|
| **CUDA GPU** | | |
| RTX 3090 (24GB) | batch=2, acc=8, 4-bit | 20-30 分钟 |
| RTX 4090 (24GB) | batch=4, acc=4, 4-bit | 15-20 分钟 |
| A100 (40GB) | batch=4, acc=4, 4-bit | 10-15 分钟 |
| **Apple Silicon** | | |
| M1 Pro (16GB) | batch=1, len=384 | 1-2 小时 |
| M2 Max (32GB) | batch=2, len=512 | 30-45 分钟 |
| M3 Max (64GB) | batch=4, len=512 | 20-30 分钟 |

*相比原数据集（21,476 样本），训练时间约为 1/20*

## 📝 CSV 数据集的优势

### 1. 快速迭代
- 样本数少，训练快
- 适合测试不同的超参数
- 快速验证想法

### 2. 易于定制
```bash
# 编辑 CSV 文件
vi ./datasets/19503488-349b-4321-941d-7875fca0737b.csv

# 或使用 Python
import pandas as pd
df = pd.read_csv('./datasets/19503488-349b-4321-941d-7875fca0737b.csv')
# 筛选、修改数据
df.to_csv('./datasets/my_custom_data.csv', index=False)
```

### 3. 组合数据集
```python
import pandas as pd

# 加载多个CSV
df1 = pd.read_csv('./datasets/data1.csv')
df2 = pd.read_csv('./datasets/data2.csv')

# 合并
combined = pd.concat([df1, df2], ignore_index=True)

# 去重
combined = combined.drop_duplicates()

# 保存
combined.to_csv('./datasets/combined.csv', index=False)
```

## 🔧 使用自己的 CSV 数据

### CSV 格式要求

必须包含以下列：
- `instruction`: 问题或指令（必需）
- `output`: 期望的回答（必需）
- `input`: 额外的输入内容（可选，可以为空）

### 创建自定义 CSV

```python
import pandas as pd

# 准备数据
data = {
    'input': ['', '', ''],  # 可以为空
    'instruction': [
        '什么是合同法？',
        '如何起诉？',
        '劳动合同纠纷怎么处理？'
    ],
    'output': [
        '合同法是规范合同关系的法律...',
        '起诉流程包括：1. 准备材料...',
        '劳动合同纠纷可以通过以下方式...'
    ]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV
df.to_csv('./datasets/my_data.csv', index=False, encoding='utf-8')
```

### 使用自定义 CSV 训练

```bash
python train.py \
  --dataset_name ./datasets/my_data.csv \
  --dataset_type csv
```

## 🎯 数据质量建议

### 1. 数据清洗
- 移除重复样本
- 检查格式是否正确
- 确保 instruction 和 output 都有内容

### 2. 数据平衡
- 不同主题的样本要均衡
- 避免某一类问题过多

### 3. 质量检查
```python
import pandas as pd

df = pd.read_csv('./datasets/19503488-349b-4321-941d-7875fca0737b.csv')

# 检查缺失值
print(df.isnull().sum())

# 检查空字符串
print((df['instruction'] == '').sum())
print((df['output'] == '').sum())

# 查看长度分布
print(df['instruction'].str.len().describe())
print(df['output'].str.len().describe())
```

## 🐛 常见问题

### Q1: CSV 文件读取错误

**错误**: `UnicodeDecodeError` 或乱码

**解决**:
```python
# 检查并转换编码
import pandas as pd

df = pd.read_csv('./datasets/data.csv', encoding='utf-8')
# 或
df = pd.read_csv('./datasets/data.csv', encoding='gbk')

# 保存为 UTF-8
df.to_csv('./datasets/data_utf8.csv', index=False, encoding='utf-8')
```

### Q2: 训练时出现 NaN 或空值

**原因**: CSV 中有缺失数据

**解决**:
```python
import pandas as pd

df = pd.read_csv('./datasets/data.csv')

# 删除有缺失值的行
df = df.dropna()

# 或填充缺失值
df['input'] = df['input'].fillna('')

df.to_csv('./datasets/data_cleaned.csv', index=False)
```

### Q3: 数据集太小效果不好

**建议**:
1. 增加训练轮数：`--num_epochs 5`
2. 使用更大的 LoRA 秩：`--lora_r 64`
3. 数据增强：同义改写、问题变换
4. 结合多个数据源

### Q4: 如何验证模型效果？

```bash
# 训练完成后，使用测试问题验证
python inference.py \
  --lora_weights ./output/llama-lora-xxx/final_model \
  --prompt "分析合同违约的法律责任" \
  --interactive
```

## 📚 参考示例

### 数据预处理脚本

```python
# preprocess_csv.py
import pandas as pd

def clean_data(csv_path):
    """清洗和预处理 CSV 数据"""
    df = pd.read_csv(csv_path)
    
    print(f"原始样本数: {len(df)}")
    
    # 删除缺失值
    df = df.dropna()
    print(f"删除缺失值后: {len(df)}")
    
    # 删除重复
    df = df.drop_duplicates(subset=['instruction', 'output'])
    print(f"删除重复后: {len(df)}")
    
    # 过滤太短的样本
    df = df[df['instruction'].str.len() > 10]
    df = df[df['output'].str.len() > 20]
    print(f"过滤短文本后: {len(df)}")
    
    # 填充空的 input
    df['input'] = df['input'].fillna('')
    
    # 打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# 使用
df = clean_data('./datasets/raw_data.csv')
df.to_csv('./datasets/cleaned_data.csv', index=False, encoding='utf-8')
print("✓ 数据清洗完成！")
```

### 数据集拆分（训练/验证）

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./datasets/19503488-349b-4321-941d-7875fca0737b.csv')

# 拆分 90% 训练，10% 验证
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv('./datasets/train.csv', index=False)
val_df.to_csv('./datasets/val.csv', index=False)

print(f"训练集: {len(train_df)} 样本")
print(f"验证集: {len(val_df)} 样本")
```

---

现在你可以使用 CSV 数据集开始训练了！建议先用小参数快速测试，确保一切正常后再进行完整训练。🚀

**快速开始**:
```bash
./train_csv.sh
```

或查看完整文档：`README.md` 和 `README_MACOS.md`

