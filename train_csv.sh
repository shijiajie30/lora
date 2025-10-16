#!/bin/bash

# CSV 数据集训练脚本

echo "======================================"
echo "Llama-3.2-3B LoRA 微调 (CSV 数据集)"
echo "======================================"

# 检查 CSV 文件是否存在
CSV_FILE="./datasets/19503488-349b-4321-941d-7875fca0737b.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "错误: CSV 文件不存在: $CSV_FILE"
    exit 1
fi

echo "✓ 找到 CSV 文件: $CSV_FILE"
echo ""

# 统计样本数
SAMPLE_COUNT=$(wc -l < "$CSV_FILE")
SAMPLE_COUNT=$((SAMPLE_COUNT - 1))  # 减去表头
echo "数据集信息:"
echo "  - 训练样本数: $SAMPLE_COUNT"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 设置环境变量（可选）
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，如需使用其他GPU请修改

# 开始训练
echo "开始训练..."
echo ""

python train.py \
    --model_name ./models/LLM-Research/Llama-3.2-3B-Instruct \
    --dataset_name "$CSV_FILE" \
    --dataset_type csv \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_length 512 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --use_4bit

echo ""
echo "======================================"
echo "训练完成！"
echo "======================================"

