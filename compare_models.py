"""
模型对比工具 - 对比微调前后的模型回答
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import platform


def parse_args():
    parser = argparse.ArgumentParser(description="对比微调前后的模型回答")
    parser.add_argument("--base_model", type=str, 
                        default="./models/LLM-Research/Llama-3.2-3B-Instruct",
                        help="基础模型路径")
    parser.add_argument("--lora_weights", type=str, required=True,
                        help="LoRA权重路径（微调后的模型）")
    parser.add_argument("--prompt", type=str, default=None,
                        help="测试问题")
    parser.add_argument("--test_file", type=str, default=None,
                        help="测试问题文件（每行一个问题）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    return parser.parse_args()


def format_prompt(text):
    """格式化为Llama指令格式"""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def load_models(base_model_path, lora_weights_path):
    """加载原始模型和微调模型"""
    
    print("=" * 80)
    print("正在加载模型...")
    print("=" * 80)
    
    # 检测设备
    is_macos = platform.system() == "Darwin"
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("✓ 使用 CUDA GPU")
    elif torch.backends.mps.is_available() and is_macos:
        device = "mps"
        dtype = torch.float32
        print("✓ 使用 Apple Silicon (MPS)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("✓ 使用 CPU")
    
    # 加载tokenizer
    print(f"\n加载 tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载原始模型（未微调）
    print(f"\n[1/2] 加载原始模型（未微调）...")
    base_model_original = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        base_model_original = base_model_original.to(device)
    base_model_original.eval()
    print("✓ 原始模型加载完成")
    
    # 加载微调模型
    print(f"\n[2/2] 加载微调模型（带LoRA）...")
    print(f"LoRA权重: {lora_weights_path}")
    base_model_finetuned = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        base_model_finetuned = base_model_finetuned.to(device)
    
    # 加载LoRA权重并合并
    finetuned_model = PeftModel.from_pretrained(base_model_finetuned, lora_weights_path)
    finetuned_model = finetuned_model.merge_and_unload()
    finetuned_model.eval()
    print("✓ 微调模型加载完成")
    
    print("\n" + "=" * 80)
    print("模型加载完成！")
    print("=" * 80 + "\n")
    
    return base_model_original, finetuned_model, tokenizer, device


def generate_response(model, tokenizer, prompt, args, device):
    """生成回答"""
    
    formatted_prompt = format_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取assistant的回复
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if assistant_marker in full_response:
        response = full_response.split(assistant_marker)[-1]
        response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
    
    return response


def compare_responses(original_model, finetuned_model, tokenizer, prompt, args, device):
    """对比两个模型的回答"""
    
    print("\n" + "=" * 80)
    print("问题:")
    print("-" * 80)
    print(prompt)
    print("=" * 80)
    
    # 原始模型回答
    print("\n[原始模型] 生成中...")
    original_response = generate_response(original_model, tokenizer, prompt, args, device)
    
    # 微调模型回答
    print("[微调模型] 生成中...\n")
    finetuned_response = generate_response(finetuned_model, tokenizer, prompt, args, device)
    
    # 显示对比
    print("=" * 80)
    print("📊 回答对比")
    print("=" * 80)
    
    print("\n🔵 原始模型回答（未微调）:")
    print("-" * 80)
    print(original_response)
    
    print("\n🟢 微调模型回答（LoRA微调后）:")
    print("-" * 80)
    print(finetuned_response)
    
    print("\n" + "=" * 80)
    
    # 简单的统计
    print("\n📈 统计信息:")
    print(f"  原始回答长度: {len(original_response)} 字符")
    print(f"  微调回答长度: {len(finetuned_response)} 字符")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    
    # 加载模型
    original_model, finetuned_model, tokenizer, device = load_models(
        args.base_model, args.lora_weights
    )
    
    # 测试问题
    test_questions = []
    
    if args.test_file:
        # 从文件读取测试问题
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_questions = [line.strip() for line in f if line.strip()]
        print(f"从文件加载了 {len(test_questions)} 个测试问题\n")
    elif args.prompt:
        test_questions = [args.prompt]
    else:
        # 默认测试问题
        test_questions = [
            "什么是合同法？",
            "劳动合同纠纷如何处理？",
            "如何起诉欠款人？"
        ]
        print("使用默认测试问题\n")
    
    # 对每个问题进行对比
    for i, question in enumerate(test_questions, 1):
        if len(test_questions) > 1:
            print(f"\n{'='*80}")
            print(f"测试 {i}/{len(test_questions)}")
            print(f"{'='*80}")
        
        compare_responses(
            original_model, 
            finetuned_model, 
            tokenizer, 
            question, 
            args, 
            device
        )
        
        if i < len(test_questions):
            input("\n按 Enter 继续下一个问题...")
    
    print("\n✓ 对比完成！")


if __name__ == "__main__":
    main()

