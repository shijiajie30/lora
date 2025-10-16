"""
推理脚本 - 使用LoRA微调后的Llama模型进行推理
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="使用微调后的模型进行推理")
    parser.add_argument("--base_model", type=str, default="./models/LLM-Research/Llama-3.2-3B-Instruct",
                        help="基础模型名称或路径")
    parser.add_argument("--lora_weights", type=str, required=True,
                        help="LoRA权重路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="输入提示词")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式对话模式")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p采样参数")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k采样参数")
    return parser.parse_args()


def load_model(base_model_path, lora_weights_path):
    """加载基础模型和LoRA权重"""
    
    print(f"正在加载基础模型: {base_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载LoRA权重
    print(f"正在加载LoRA权重: {lora_weights_path}")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model = model.merge_and_unload()  # 合并LoRA权重到基础模型
    
    model.eval()
    
    return model, tokenizer


def format_prompt(text):
    """将用户输入格式化为Llama指令格式"""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def generate_response(model, tokenizer, prompt, args):
    """生成模型回复"""
    
    # 格式化输入
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取assistant的回复
    # 查找assistant标签后的内容
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if assistant_marker in full_response:
        response = full_response.split(assistant_marker)[-1]
        # 移除结束标记
        response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入部分
        if prompt in response:
            response = response.split(prompt)[-1].strip()
    
    return response


def interactive_mode(model, tokenizer, args):
    """交互式对话模式"""
    
    print("\n" + "=" * 60)
    print("进入交互式对话模式")
    print("输入 'exit' 或 'quit' 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("退出对话")
                break
            
            if not user_input:
                continue
            
            print("\n正在生成回复...\n")
            response = generate_response(model, tokenizer, user_input, args)
            print(f"助手: {response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n退出对话")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue


def single_inference(model, tokenizer, prompt, args):
    """单次推理"""
    
    print(f"\n用户: {prompt}\n")
    print("正在生成回复...\n")
    
    response = generate_response(model, tokenizer, prompt, args)
    
    print(f"助手: {response}\n")


def main():
    args = parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.base_model, args.lora_weights)
    
    print("\n模型加载完成！\n")
    
    # 根据模式运行
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        single_inference(model, tokenizer, args.prompt, args)
    else:
        print("请使用 --prompt 指定提示词，或使用 --interactive 进入交互模式")
        print("示例:")
        print("  python inference.py --lora_weights ./output/llama-lora-xxx/final_model --prompt '请解释合同法的基本原则'")
        print("  python inference.py --lora_weights ./output/llama-lora-xxx/final_model --interactive")


if __name__ == "__main__":
    main()

