from mixed_precision_policy import generate_mixed_precision_config

def test_policy():
    num_layers = 22
    default_bits = 5
    high_bits = 8
    
    config = generate_mixed_precision_config(num_layers, default_bits, high_bits)
    
    print(f"Generated config for {num_layers} layers.")
    print(f"Total keys: {len(config)}")
    
    # Check first layer (Sensitive)
    print("\nChecking Layer 0 (Sensitive):")
    for k, v in config.items():
        if "layers.0." in k:
            print(f"{k}: {v}")
            assert v == high_bits, f"Layer 0 should be high precision, got {v}"

    # Check middle layer (Layer 10)
    print("\nChecking Layer 10 (Middle):")
    for k, v in config.items():
        if "layers.10." in k:
            print(f"{k}: {v}")
            if "o_proj" in k or "gate_proj" in k:
                assert v == high_bits, f"Layer 10 sensitive module should be high precision, got {v}"
            else:
                assert v == default_bits, f"Layer 10 robust module should be default precision, got {v}"

    # Check last layer (Sensitive)
    print("\nChecking Layer 21 (Sensitive):")
    for k, v in config.items():
        if "layers.21." in k:
            print(f"{k}: {v}")
            assert v == high_bits, f"Layer 21 should be high precision, got {v}"
            
    print("\nPolicy test passed!")

if __name__ == "__main__":
    test_policy()
