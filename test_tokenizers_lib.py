from tokenizers import Tokenizer
import json

model_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix/block_1x128_mantissa_3/tokenizer.json"

try:
    print(f"Loading tokenizer from {model_path}")
    tokenizer = Tokenizer.from_file(model_path)
    print("Successfully loaded tokenizer")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Let's also try to manually parse json and see if we can spot anything
try:
    with open(model_path, 'r') as f:
        data = json.load(f)
    print("JSON is valid")
    print("Model type:", data['model']['type'])
    print("Merges type:", type(data['model']['merges']))
    if len(data['model']['merges']) > 0:
        print("First merge:", data['model']['merges'][0])
except Exception as e:
    print(f"JSON error: {e}")
