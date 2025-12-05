def generate_mixed_precision_config(num_layers, default_bits=5, high_precision_bits=8):
    """
    Generates a layer-wise mixed precision configuration for TinyLlama.
    
    Strategy:
    1. First 2 and Last 2 layers: High precision (high_precision_bits) for all modules.
    2. Middle layers:
       - Sensitive modules (o_proj, gate_proj): High precision.
       - Robust modules (q_proj, k_proj, v_proj, up_proj, down_proj): Default precision (default_bits).
    
    Args:
        num_layers (int): Total number of transformer layers.
        default_bits (int): Bit width for robust layers (e.g., 4 or 5).
        high_precision_bits (int): Bit width for sensitive layers (e.g., 8).
        
    Returns:
        dict: A dictionary mapping module names (suffix) to mantissa bits.
              Note: Since LlamaMyModel applies quantization layer-by-layer, 
              we might need to structure this by layer index.
    """
    config = {}
    
    # Define sensitive layers (indices)
    sensitive_layer_indices = set([0, 1, num_layers-2, num_layers-1])
    
    # Define module types
    all_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                   'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    
    sensitive_modules = set(['self_attn.o_proj', 'mlp.gate_proj'])
    
    for i in range(num_layers):
        for module in all_modules:
            key = f"model.layers.{i}.{module}"
            
            if i in sensitive_layer_indices:
                bits = high_precision_bits
            else:
                # Middle layers
                if module in sensitive_modules:
                    bits = high_precision_bits
                else:
                    bits = default_bits
            
            config[key] = bits
            
    return config
