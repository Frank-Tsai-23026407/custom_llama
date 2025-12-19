#!/usr/bin/env python3
"""
Convert models in the model directory to BF16 format and delete originals.
This script processes base models and saves them in BF16 to reduce storage by ~50%.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
from pathlib import Path
import argparse

def get_model_size_mb(model_path):
    """Calculate total size of model directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def convert_model_to_bf16(model_path, temp_suffix="_bf16_temp"):
    """
    Convert a model to BF16 format and replace the original.
    
    Args:
        model_path: Path to the model directory
        temp_suffix: Suffix for temporary directory during conversion
    """
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    temp_path = model_path + temp_suffix
    
    try:
        # Get original size
        original_size_mb = get_model_size_mb(model_path)
        print(f"\n{'='*80}")
        print(f"Converting: {model_path}")
        print(f"Original size: {original_size_mb:.2f} MB ({original_size_mb/1024:.2f} GB)")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Load in FP32 first
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Convert to BF16
        print("Converting to BF16...")
        model = model.to(dtype=torch.bfloat16)
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save to temporary directory
        print(f"Saving BF16 model to temporary location...")
        model.save_pretrained(temp_path, safe_serialization=True)
        tokenizer.save_pretrained(temp_path)
        
        # Get new size
        new_size_mb = get_model_size_mb(temp_path)
        savings_mb = original_size_mb - new_size_mb
        savings_pct = (savings_mb / original_size_mb) * 100
        
        print(f"BF16 size: {new_size_mb:.2f} MB ({new_size_mb/1024:.2f} GB)")
        print(f"Savings: {savings_mb:.2f} MB ({savings_mb/1024:.2f} GB, {savings_pct:.1f}%)")
        
        # Delete original
        print("Deleting original model...")
        shutil.rmtree(model_path)
        
        # Rename temporary to original
        print("Renaming BF16 model to original path...")
        os.rename(temp_path, model_path)
        
        print(f"✅ Successfully converted {model_path} to BF16")
        print(f"{'='*80}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {model_path}: {e}")
        # Cleanup temp directory if it exists
        if os.path.exists(temp_path):
            print(f"Cleaning up temporary directory: {temp_path}")
            shutil.rmtree(temp_path)
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert quantized models to BF16 format")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing models to convert (default: models)")
    args = parser.parse_args()
    
    base_dir = Path(args.model_dir)
    
    # Find all quantized model directories
    models_to_convert = []
    for item in base_dir.iterdir():
        if item.is_dir() and ("awq" in item.name or "bfp" in item.name):
            models_to_convert.append(str(item))
    
    models_to_convert.sort()
    
    print("="*80)
    print("Model to BF16 Conversion Script")
    print("="*80)
    print("\nThis script will:")
    print("1. Convert each model to BF16 format")
    print("2. Delete the original FP32 model files")
    print("3. Save ~50% storage space per model")
    print("\nModels to convert:")
    for model_path in models_to_convert:
        print(f"  - {model_path}")
    
    # Calculate total original size
    total_original_mb = sum(get_model_size_mb(p) for p in models_to_convert if os.path.exists(p))
    print(f"\nTotal original size: {total_original_mb:.2f} MB ({total_original_mb/1024:.2f} GB)")
    
    response = input("\n⚠️  Proceed with conversion? Original models will be deleted! [y/N]: ")
    if response.lower() != 'y':
        print("Conversion cancelled.")
        return
    
    # Convert models
    results = []
    for model_path in models_to_convert:
        success = convert_model_to_bf16(model_path)
        results.append((model_path, success))
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if successful > 0:
        total_new_mb = sum(get_model_size_mb(p) for p, success in results if success and os.path.exists(p))
        total_savings_mb = total_original_mb - total_new_mb
        total_savings_pct = (total_savings_mb / total_original_mb) * 100
        
        print(f"\nTotal new size: {total_new_mb:.2f} MB ({total_new_mb/1024:.2f} GB)")
        print(f"Total savings: {total_savings_mb:.2f} MB ({total_savings_mb/1024:.2f} GB, {total_savings_pct:.1f}%)")
    
    print("\n✅ Conversion complete!")
    print("="*80)

if __name__ == "__main__":
    main()
