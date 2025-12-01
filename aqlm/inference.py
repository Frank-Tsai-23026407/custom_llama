import argparse
import torch
from transformers import AutoTokenizer
from src.modelutils import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to original model")
    parser.add_argument("--quantized_path", type=str, required=True, help="Path to quantized checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path} with quantized weights from {args.quantized_path}")
    # load_quantized argument in get_model triggers loading of quantized weights
    model = get_model(args.model_path, load_quantized=args.quantized_path, dtype="auto", device_map="cpu")

    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)

    print(f"Generating for prompt: '{args.prompt}'")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("-" * 50)
    print(output_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
