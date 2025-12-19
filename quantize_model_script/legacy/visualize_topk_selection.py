"""
Visual demonstration of top-k selection in AWQ Mix-Precision quantization.

This script shows exactly how top-k entries are selected within a 2D block.
"""

import torch
import numpy as np


def visualize_topk_selection():
    """Demonstrate top-k selection with a small example."""
    
    print("=" * 80)
    print("Top-K Selection Visualization for AWQ Mix-Precision")
    print("=" * 80)
    
    # Create a small example block
    torch.manual_seed(42)
    np.random.seed(42)
    
    block_height, block_width = 4, 4
    weight_block = torch.randn(block_height, block_width) * 0.5
    
    # Simulate importance scores (per column/input feature)
    importance_scores = torch.tensor([0.5, 1.2, 0.8, 1.5])
    
    # Calculate weighted weights (salience)
    weighted_block = weight_block * importance_scores.unsqueeze(0)
    salience = torch.abs(weighted_block)
    
    print(f"\nExample: {block_height}×{block_width} block with top_k=3")
    print("=" * 80)
    
    print("\n[1] Original Weight Block:")
    print("-" * 40)
    for i in range(block_height):
        row_str = "  "
        for j in range(block_width):
            row_str += f"{weight_block[i,j]:7.3f} "
        print(row_str)
    
    print("\n[2] Importance Scores (per input feature/column):")
    print("-" * 40)
    print("  ", end="")
    for score in importance_scores:
        print(f"{score:7.3f} ", end="")
    print()
    
    print("\n[3] Weighted Block (weight × importance):")
    print("-" * 40)
    for i in range(block_height):
        row_str = "  "
        for j in range(block_width):
            row_str += f"{weighted_block[i,j]:7.3f} "
        print(row_str)
    
    print("\n[4] Salience (absolute value of weighted block):")
    print("-" * 40)
    for i in range(block_height):
        row_str = "  "
        for j in range(block_width):
            row_str += f"{salience[i,j]:7.3f} "
        print(row_str)
    
    # Find top-k
    top_k = 3
    salience_flat = salience.flatten()
    topk_values, topk_indices_flat = torch.topk(salience_flat, top_k)
    
    # Convert flat indices to 2D positions
    topk_positions = []
    for idx in topk_indices_flat:
        row = idx.item() // block_width
        col = idx.item() % block_width
        topk_positions.append((row, col))
    
    print(f"\n[5] Top-{top_k} Selection (highest salience):")
    print("-" * 40)
    print(f"  Total entries in block: {block_height * block_width}")
    print(f"  Selecting top-{top_k} entries ({top_k/(block_height*block_width)*100:.1f}%)")
    print()
    
    for rank, ((row, col), value) in enumerate(zip(topk_positions, topk_values), 1):
        original_value = weight_block[row, col].item()
        salience_value = value.item()
        print(f"  Rank {rank}: Position [{row},{col}]")
        print(f"           Original weight: {original_value:7.3f}")
        print(f"           Salience:        {salience_value:7.3f}")
        print()
    
    print("[6] Visual Map of Selection:")
    print("-" * 40)
    print("  Legend: 'F' = FP32 (selected), 'B' = BFP (quantized)")
    print()
    
    selection_map = [['B' for _ in range(block_width)] for _ in range(block_height)]
    for row, col in topk_positions:
        selection_map[row][col] = 'F'
    
    for i in range(block_height):
        row_str = "  "
        for j in range(block_width):
            marker = selection_map[i][j]
            value = weight_block[i,j].item()
            if marker == 'F':
                row_str += f"[{marker} {value:5.2f}] "
            else:
                row_str += f" {marker} {value:5.2f}  "
        print(row_str)
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    print(f"""
1. Top-k selection is PER BLOCK:
   - We have 1 block of size {block_height}×{block_width} = {block_height*block_width} entries
   - We select {top_k} entries from these {block_height*block_width} total entries
   
2. Selection is based on SALIENCE (not just weight magnitude):
   - Salience = |weight × importance_score|
   - Higher importance_score → higher chance of selection
   
3. Selected entries can be SCATTERED:
   - The {top_k} entries are at positions: {topk_positions}
   - They are NOT confined to a single row or column
   - They are distributed based on salience across the entire block
   
4. What happens to each entry:
   - {top_k} entries (marked 'F'): Kept in full FP32 precision
   - {block_height*block_width - top_k} entries (marked 'B'): Quantized to BFP (e.g., 4-bit mantissa)
   
5. Final representation:
   - Hybrid format: mix of FP32 and BFP values
   - Storage overhead: {top_k} FP32 values + {block_height*block_width - top_k} BFP values + 1 shared exponent
""")
    
    print("=" * 80)


def demonstrate_different_topk_values():
    """Show how different top-k values affect selection."""
    
    print("\n\n")
    print("=" * 80)
    print("Effect of Different Top-K Values")
    print("=" * 80)
    
    block_size = 16  # 4×4 block
    
    print(f"\nFor a 4×4 block ({block_size} total entries):\n")
    print(f"{'top_k':<10} {'FP32 Entries':<15} {'BFP Entries':<15} {'FP32 %':<10}")
    print("-" * 50)
    
    for k in [1, 4, 8, 12, 16]:
        fp32_count = k
        bfp_count = block_size - k
        fp32_percent = (k / block_size) * 100
        print(f"{k:<10} {fp32_count:<15} {bfp_count:<15} {fp32_percent:<10.1f}%")
    
    print("\nTypical settings:")
    print("  - top_k=16 for 16×16 blocks (256 entries) → 6.25% FP32")
    print("  - top_k=8  for 8×8 blocks (64 entries)   → 12.5% FP32")
    print("  - top_k=4  for 4×4 blocks (16 entries)   → 25.0% FP32")
    print("\nTrade-off:")
    print("  - Higher top_k: Better accuracy, more FP32 storage")
    print("  - Lower top_k:  Less accuracy, more BFP compression")
    

if __name__ == "__main__":
    visualize_topk_selection()
    demonstrate_different_topk_values()
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
