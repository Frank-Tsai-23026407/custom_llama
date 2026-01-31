from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf"
save_dir = "./aqlm/model/TinyLlama-v1.0-hf-aqlm"

def download_model():
    print(f"Starting download of {model_id}...")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Download and save tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)
    
    # Download and save model
    print("Downloading model weights (this may take a few minutes)...")
    # Using trust_remote_code=True as AQLM models often require it
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        cache_dir=save_dir  # This will help organize the download
    )
    
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    
    print("\nDownload complete!")
    print(f"Model and tokenizer are saved in: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    download_model()
