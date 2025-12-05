import torch
import torch.nn as nn
from maddness_hadamard import MaddnessHadamardLayer

def test_layer():
    print("Testing MaddnessHadamardLayer...")
    in_features = 1024
    out_features = 1024
    original_layer = nn.Linear(in_features, out_features)
    
    # Test Level 1
    print("Testing Level 1...")
    layer = MaddnessHadamardLayer(original_layer, subspace_dim=256, num_levels=1)
    x = torch.randn(10, in_features)
    y = layer(x)
    print(f"Forward pass successful. Output shape: {y.shape}")
    
    # Test Training
    print("Testing Training...")
    X_calib = torch.randn(100, in_features)
    W = original_layer.weight.data
    mse, max_diff = layer.train_and_configure(X_calib, W)
    print(f"Training successful. MSE: {mse}, Max Diff: {max_diff}")
    
    # Test Level 2
    print("\nTesting Level 2...")
    layer_l2 = MaddnessHadamardLayer(original_layer, subspace_dim=256, num_levels=2)
    mse, max_diff = layer_l2.train_and_configure(X_calib, W)
    print(f"Training L2 successful. MSE: {mse}, Max Diff: {max_diff}")
    
    y_l2 = layer_l2(x)
    print(f"Forward pass L2 successful. Output shape: {y_l2.shape}")
    
    # Check if parameters are registered correctly
    print(f"Split indices shape: {layer_l2.split_indices.shape}")
    assert layer_l2.split_indices.shape[0] == 2
    
    print("All tests passed!")

if __name__ == "__main__":
    test_layer()
