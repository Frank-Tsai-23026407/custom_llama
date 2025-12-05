import random
import time
from utils import register_generation_function

@register_generation_function("Random Word")
def random_word_generate(model_path, history_transformer_format):
    # This function doesn't need a model!
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    
    # Simulate streaming
    response = f"Here is a random word: {random.choice(words)}"
    partial = ""
    for char in response:
        partial += char
        time.sleep(0.05) # Simulate typing speed
        yield partial
