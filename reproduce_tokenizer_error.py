from transformers import AutoTokenizer
import sys

model_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix/block_1x128_mantissa_3"

try:
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Successfully loaded tokenizer")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
