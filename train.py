"""
LoRA微调脚本 - Llama-3.2-3B-Instruct
使用Skepsun/lawyer_llama_data数据集进行法律领域微调
"""

import os
import torch
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA微调Llama模型")
    parser.add_argument("--model_name", type=str, default="./models/LLM-Research/Llama-3.2-3B-Instruct",
                        help="预训练模型名称或路径")
    parser.add_argument("--dataset_name", type=str, default="./datasets/19503488-349b-4321-941d-7875fca0737b.csv",
                        help="数据集名称（HuggingFace）或CSV文件路径")
    parser.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "huggingface", "csv"],
                        help="数据集类型：auto（自动检测）、huggingface、csv")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="模型输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="训练批次大小（3B模型建议使用1-2）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数（3B模型建议增加以保持有效批次大小）")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率（3B模型建议使用更小的学习率）")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA秩（3B模型可以使用更大的秩）")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="使用4bit量化")
    parser.add_argument("--use_flash_attention", action="store_true", default=False,
                        help="使用Flash Attention 2")
    return parser.parse_args()


def load_and_prepare_model(args):
    """加载并准备模型进行LoRA微调"""
    
    print(f"正在加载模型: {args.model_name}")
    
    # 检查是否在 macOS 上运行
    import platform
    is_macos = platform.system() == "Darwin"
    
    # 检测可用的设备
    if torch.cuda.is_available():
        device = "cuda"
        print("✓ 检测到 CUDA GPU")
    elif torch.backends.mps.is_available() and is_macos:
        device = "mps"
        print("✓ 检测到 Apple Silicon (MPS)")
        # macOS 上禁用 4-bit 量化
        if args.use_4bit:
            print("⚠️  macOS 不支持 4-bit 量化，已自动禁用")
            args.use_4bit = False
    else:
        device = "cpu"
        print("⚠️  使用 CPU 训练（速度较慢）")
        if args.use_4bit:
            print("⚠️  CPU 不支持 4-bit 量化，已自动禁用")
            args.use_4bit = False
    
    # 配置量化（仅在 CUDA 上可用）
    if args.use_4bit and torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("✓ 启用 4-bit 量化")
        except Exception as e:
            print(f"⚠️  量化配置失败: {e}")
            bnb_config = None
    else:
        bnb_config = None
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    # 根据设备选择合适的数据类型
    if device == "mps":
        # MPS 对 bfloat16 支持有限，使用 float32
        model_dtype = torch.float32
        print("✓ 使用 float32 精度（MPS 最佳兼容）")
    elif device == "cpu":
        model_dtype = torch.float32
        print("✓ 使用 float32 精度（CPU）")
    else:
        model_dtype = torch.bfloat16
        print("✓ 使用 bfloat16 精度")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.use_4bit else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        dtype=model_dtype,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "sdpa"
    )
    
    # 如果不是 CUDA，手动移动到设备
    if device != "cuda" and not args.use_4bit:
        model = model.to(device)
        print(f"✓ 模型已加载到 {device}")
    
    # 准备模型进行k-bit训练（仅 CUDA + 量化）
    if args.use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_and_prepare_dataset(args, tokenizer):
    """加载并预处理数据集"""
    
    print(f"正在加载数据集: {args.dataset_name}")
    
    # 自动检测数据集类型
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        if args.dataset_name.endswith('.csv'):
            dataset_type = "csv"
            print("✓ 自动检测到 CSV 格式")
        elif os.path.exists(args.dataset_name):
            dataset_type = "local"
            print("✓ 自动检测到本地数据集")
        else:
            dataset_type = "huggingface"
            print("✓ 使用 HuggingFace 数据集")
    
    # 根据类型加载数据集
    if dataset_type == "csv":
        # 从 CSV 文件加载
        if not os.path.exists(args.dataset_name):
            raise FileNotFoundError(f"CSV 文件不存在: {args.dataset_name}")
        
        print(f"从 CSV 文件加载数据...")
        dataset = load_dataset("csv", data_files={"train": args.dataset_name})
        print(f"✓ 成功加载 {len(dataset['train'])} 个训练样本")
        
    elif dataset_type == "local" or os.path.exists(args.dataset_name):
        # 本地路径，直接指向 arrow 文件所在目录
        dataset = load_dataset("arrow", data_files={
            "train": os.path.join(args.dataset_name, "*.arrow")
        })
        print(f"✓ 从本地加载 {len(dataset['train'])} 个样本")
        
    else:
        # HuggingFace 数据集名称，会自动查找缓存目录
        dataset = load_dataset(args.dataset_name, cache_dir="./datasets")
        print(f"✓ 从 HuggingFace 加载 {len(dataset['train'])} 个样本")
    
    # 定义对话格式化函数
    def format_instruction(sample):
        """将数据格式化为Llama的指令格式"""
        if "instruction" in sample and "output" in sample:
            # 格式1: instruction-output
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            output = sample["output"]
            
            if input_text:
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        
        elif "question" in sample and "answer" in sample:
            # 格式2: question-answer
            question = sample["question"]
            answer = sample["answer"]
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        
        elif "conversations" in sample:
            # 格式3: 多轮对话
            conversations = sample["conversations"]
            prompt = "<|begin_of_text|>"
            for conv in conversations:
                role = "user" if conv["from"] == "human" else "assistant"
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{conv['value']}<|eot_id|>"
        
        else:
            # 默认格式：直接使用文本
            prompt = sample.get("text", str(sample))
        
        return prompt
    
    # 预处理函数
    def preprocess_function(examples):
        # 格式化文本
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            sample = {key: examples[key][i] for key in examples.keys()}
            texts.append(format_instruction(sample))
        
        # Tokenize - 使用padding到最大长度以确保批次一致性
        model_inputs = tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            padding="max_length",  # 填充到最大长度
            return_tensors=None,
        )
        
        # 设置labels（对于因果语言模型，labels就是input_ids）
        # 需要深拷贝，否则会共享同一个列表
        model_inputs["labels"] = [ids[:] for ids in model_inputs["input_ids"]]
        
        return model_inputs
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )
    
    return tokenized_dataset


def train(args):
    """主训练函数"""
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"llama-lora-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    model, tokenizer = load_and_prepare_model(args)
    
    # 加载数据集
    tokenized_dataset = load_and_prepare_dataset(args, tokenizer)
    
    # 检测设备并配置训练参数
    import platform
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available() and platform.system() == "Darwin"
    
    # 根据设备选择优化器和精度
    if is_cuda and args.use_4bit:
        optim = "paged_adamw_8bit"
        use_bf16 = True
        use_fp16 = False
        print("✓ 使用 paged_adamw_8bit 优化器 + bf16")
    elif is_cuda:
        optim = "adamw_torch"
        use_bf16 = True
        use_fp16 = False
        print("✓ 使用 adamw_torch 优化器 + bf16")
    else:
        # MPS 或 CPU
        optim = "adamw_torch"
        use_bf16 = False
        use_fp16 = False
        print("✓ 使用 adamw_torch 优化器 + fp32")
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        group_by_length=True,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        use_mps_device=is_mps,  # 启用 MPS 加速
    )
    
    # 数据整理器
    # 由于已经在预处理阶段完成了padding，使用default_data_collator即可
    data_collator = default_data_collator
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n训练完成！模型已保存到: {final_model_path}")
    print(f"LoRA适配器权重已保存")
    
    return final_model_path


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LoRA微调 - Llama-3.2-3B-Instruct")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.dataset_name}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"学习率: {args.learning_rate}")
    print(f"LoRA参数: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"4-bit量化: {args.use_4bit}")
    print("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()

