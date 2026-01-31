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
from custom_model_only_quantize.runtime_quantize import QuantizationConfig


# --- Core Generation Functions ---

@register_generation_function("Hugging Face")
def hf_generate(model_path, history_transformer_format, quantization_config=None):
    model, tokenizer = get_or_load_model(model_path, "hf", quantization_config)
    
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
def maddness_generate(model_path, history_transformer_format, quantization_config=None):
    # Maddness doesn't support runtime quantization in this context yet
    model, tokenizer = get_or_load_model(model_path, "maddness_hadamard", None)
    
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
def llama_my_generate(model_path, history_transformer_format, quantization_config=None):
    model, tokenizer = get_or_load_model(model_path, "custom", quantization_config)
    
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    
    # Using blocking generate for Custom backend
    outputs = model.generate(messages, max_new_tokens=1024)
    
    input_ids = tokenizer([messages], return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    new_tokens = outputs[0][prompt_len:]
    
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    yield decoded_output
