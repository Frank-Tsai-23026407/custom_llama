import json
from tokenizers import Tokenizer

model_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix/block_1x128_mantissa_3/tokenizer.json"
patched_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix/block_1x128_mantissa_3/tokenizer_patched.json"

try:
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    merges = data['model']['merges']
    if len(merges) > 0 and isinstance(merges[0], list):
        print("Converting merges from list of lists to list of strings...")
        new_merges = [f"{m[0]} {m[1]}" for m in merges]
        data['model']['merges'] = new_merges
        
        with open(patched_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved patched tokenizer to {patched_path}")
        
        print("Attempting to load patched tokenizer...")
        tokenizer = Tokenizer.from_file(patched_path)
        print("Successfully loaded patched tokenizer!")
    else:
        print("Merges are not list of lists, skipping patch.")

except Exception as e:
    print(f"Error: {e}")
