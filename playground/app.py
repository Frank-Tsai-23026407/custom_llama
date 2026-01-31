import os
import sys

# Set CUDA environment variables for JIT compilation
cuda_home = "/home/frank23026407/miniconda3"
os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
os.environ["CPATH"] = f"{cuda_home}/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6" # Set for RTX 30/40 series, A100, etc.

import gradio as gr
import torch
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
from custom_model_only_quantize.runtime_quantize import QuantizationConfig

# Import custom functions to ensure they are registered
import basic_functions
import custom_functions_example


# --- Main Predict Function ---

def predict(message, history, model_path, backend_name, quantization_mode, block_height, block_width, mantissa_bits, activation_file, calibration_dataset):
    history_transformer_format = history + [[message, ""]]
    
    if backend_name not in GENERATION_FUNCTIONS:
        yield f"Error: Backend '{backend_name}' not found."
        return

    # Map quantization mode to config
    quantization_config = None
    if quantization_mode != "None":
        activations = None
        
        # Check source: File or Dataset
        if "AWQ" in quantization_mode:
            if calibration_dataset != "Local File":
                 # Will be handled in utils.py
                 pass
            elif activation_file:
                try:
                    # Assuming activations are saved as a PyTorch dictionary
                    activations = torch.load(activation_file, map_location=device)
                    if not isinstance(activations, dict):
                        yield f"Error: Activation file does not contain a dictionary."
                        return
                    # Note: We don't perform calibration here, we just load.
                except Exception as e:
                    yield f"Error loading activations from '{activation_file}': {str(e)}"
                    return
            else:
                yield "Error: AWQ selected but no Activation File or Calibration Dataset provided."
                return

        method = "bfp"
        if "AWQ Fix" in quantization_mode:
            method = "awq-fix"
        elif "AWQ Mix" in quantization_mode:
            method = "awq-mix"
            
        quantization_config = QuantizationConfig(
            method=method, 
            block_height=int(block_height), 
            block_width=int(block_width), 
            mantissa_bits=int(mantissa_bits),
            activations=activations,
            calibration_dataset=calibration_dataset
        )

    gen_func = GENERATION_FUNCTIONS[backend_name]
    
    try:
        for response in gen_func(model_path, history_transformer_format, quantization_config):
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
        quantization_input = gr.Dropdown(
            label="Quantization (Runtime)", 
            choices=["None", "Runtime BFP (Dryrun)", "Runtime AWQ Fix (Dryrun)", "Runtime AWQ Mix (Dryrun)"], 
            value="None",
            scale=1
        )
    
    # Quantization Parameters Group
    with gr.Column(visible=False) as quant_group:
        quant_markdown = gr.Markdown("### Quantization Parameters (only for Runtime)")
        with gr.Row() as bfp_params_row:
            block_height_input = gr.Number(label="Block Height", value=16, precision=0, scale=1)
            block_width_input = gr.Number(label="Block Width", value=16, precision=0, scale=1)
            mantissa_bits_input = gr.Number(label="Mantissa Bits", value=4, precision=0, scale=1)
        
        with gr.Column(visible=False) as calibration_group: # Initially hidden
            calibration_markdown = gr.Markdown("### Calibration Source (For AWQ)")
            with gr.Row() as calibration_row:
                calibration_dataset_input = gr.Dropdown(
                    label="Calibration Source",
                    choices=["Local File", "wikitext2", "c4", "ptb"],
                    value="Local File",
                    scale=1
                )
                
                # Activation File for AWQ
                activation_file_input = gr.Textbox(
                    label="Activation File Path (Pre-computed)", 
                    placeholder="/path/to/activations.pt", 
                    visible=True,
                    scale=2
                )

    # dynamic change visibility
    def on_quantization_change(mode):
        is_visible = (mode != "None")
        is_awq = ("AWQ" in mode)
        
        # Explicitly hide/show from inside out
        # If invisible, hide everything.
        # If visible:
        #   - bfp row is always visible
        #   - calibration row is visible only if AWQ
        
        return {
            quant_markdown: gr.update(visible=is_visible),
            bfp_params_row: gr.update(visible=is_visible),
            calibration_row: gr.update(visible=is_visible and is_awq),
            calibration_markdown: gr.update(visible=is_visible and is_awq),
            calibration_group: gr.update(visible=is_visible and is_awq),
            quant_group: gr.update(visible=is_visible),
        }

    def on_dataset_change(dataset):
        # only show activation_file_input if dataset is Local File
        return {
            activation_file_input: gr.update(visible=(dataset == "Local File"))
        }

    # bind events
    quantization_input.change(
        fn=on_quantization_change,
        inputs=[quantization_input],
        outputs=[quant_markdown, bfp_params_row, calibration_row, calibration_markdown, calibration_group, quant_group]
    )

    # bind events
    calibration_dataset_input.change(
        fn=on_dataset_change,
        inputs=[calibration_dataset_input],
        outputs=[activation_file_input]
    )

    chat_interface = gr.ChatInterface(
        predict,
        additional_inputs=[
            model_path_input, 
            backend_input, 
            quantization_input,
            block_height_input,
            block_width_input,
            mantissa_bits_input,
            activation_file_input,
            calibration_dataset_input
        ],
        title=None,
        description="Ask Tiny llama any questions. Change model/backend to reload.",
        examples=[
            ['How to cook a fish?', "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Hugging Face", "None", 16, 16, 4, "", "Local File"], 
            ['Who is the president of US now?', "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Llama Custom", "Runtime BFP (Dryrun)", 32, 32, 4, "", "Local File"],
        ]
    )

if __name__ == "__main__":
    demo.launch()
