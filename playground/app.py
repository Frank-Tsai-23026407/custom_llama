import gradio as gr
import torch
import sys
from pathlib import Path
from threading import Thread
from transformers import TextIteratorStreamer, StoppingCriteriaList

# Import from our new utils module
# This handles sys.path setup and imports LlamaMyModel, etc.
from utils import (
    register_generation_function, 
    get_or_load_model, 
    GENERATION_FUNCTIONS, 
    StopOnTokens, 
    device
)

# Import custom functions to ensure they are registered
import custom_functions

# --- Core Generation Functions ---

@register_generation_function("Hugging Face")
def hf_generate(model_path, history_transformer_format):
    model, tokenizer = get_or_load_model(model_path, "hf")
    
    stop = StopOnTokens()
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:
            break
        yield partial_message

@register_generation_function("Maddness Hadamard")
def maddness_generate(model_path, history_transformer_format):
    model, tokenizer = get_or_load_model(model_path, "maddness_hadamard")
    
    stop = StopOnTokens()
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:
            break
        yield partial_message

@register_generation_function("Llama Custom")
def llama_my_generate(model_path, history_transformer_format):
    model, tokenizer = get_or_load_model(model_path, "custom")
    
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    
    # Using blocking generate for Custom backend
    outputs = model.generate(messages, max_new_tokens=1024)
    
    input_ids = tokenizer([messages], return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    new_tokens = outputs[0][prompt_len:]
    
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    yield decoded_output

# --- Main Predict Function ---

def predict(message, history, model_path, backend_name):
    history_transformer_format = history + [[message, ""]]
    
    if backend_name not in GENERATION_FUNCTIONS:
        yield f"Error: Backend '{backend_name}' not found."
        return

    gen_func = GENERATION_FUNCTIONS[backend_name]
    
    try:
        for response in gen_func(model_path, history_transformer_format):
            yield response
    except Exception as e:
        yield f"Error during generation: {str(e)}"
        import traceback
        traceback.print_exc()

# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# Tinyllama ChatBot Playground")
    
    with gr.Row():
        model_path_input = gr.Textbox(
            label="Model Path", 
            value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            scale=2
        )
        # Dynamically populate choices
        backend_input = gr.Dropdown(
            label="Backend", 
            choices=list(GENERATION_FUNCTIONS.keys()), 
            value="Hugging Face",
            scale=1
        )

    chat_interface = gr.ChatInterface(
        predict,
        additional_inputs=[model_path_input, backend_input],
        title=None,
        description="Ask Tiny llama any questions. Change model/backend to reload.",
        examples=[
            ['How to cook a fish?', "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Hugging Face"], 
            ['Who is the president of US now?', "model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix/block_1x128_mantissa_3", "Llama Custom"],
            ['Give me a word', "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Random Word"],
            ['Tell me a story', "maddness_hadamard_tinyllama_512_d4_l1.pt", "Maddness Hadamard"]
        ]
    )

if __name__ == "__main__":
    demo.launch()
