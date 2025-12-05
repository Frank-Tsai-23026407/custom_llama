import torch
import numpy as np
import matplotlib.pyplot as plt

def _get_hadamard_unnormalized(n, device='cpu'):
    """
    Recursively generates an unnormalized Hadamard matrix of size n x n.

    Args:
        n (int): The size of the matrix. Must be a power of 2.
        device (str or torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: The unnormalized Hadamard matrix.
    """
    if n == 1:
        return torch.ones(1, 1, device=device)
    h_n_2 = _get_hadamard_unnormalized(n // 2, device=device)
    top = torch.cat([h_n_2, h_n_2], dim=1)
    bottom = torch.cat([h_n_2, -h_n_2], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_hadamard_matrix(n, device='cpu'):
    """
    Generates a normalized Hadamard matrix of size n x n.
    The matrix is normalized by dividing by sqrt(n) so that it is orthogonal.

    Args:
        n (int): The size of the matrix. Must be a power of 2.
        device (str or torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: The normalized Hadamard matrix.
    """
    H = _get_hadamard_unnormalized(n, device)
    return H / torch.sqrt(torch.tensor(n, dtype=torch.float32, device=device))

def apply_block_hadamard(x, H_block):
    """
    Applies block diagonal Hadamard transform.
    x: [..., dim]
    H_block: [B, B]
    """
    B = H_block.shape[0]
    dim = x.shape[-1]
    if dim % B != 0:
        return x # Should not happen if checked in init
    
    original_shape = x.shape
    num_blocks = dim // B
    x = x.view(*original_shape[:-1], num_blocks, B)
    x = x @ H_block.t()
    return x.view(*original_shape)

def plot_3d_activations(activations, tokens, title, save_path, z_lim=None):
    """
    Plots activations in a 3D visualization where:
    - X-axis: Tokens (Sequence dimension)
    - Y-axis: Channels (Feature dimension)
    - Z-axis: Activation Value
    
    Args:
        activations (torch.Tensor): Tensor of shape (seq_len, hidden_dim).
        tokens (list): List of token strings corresponding to the sequence.
        title (str): Title of the plot.
        save_path (str): File path to save the plot image.
        z_lim (tuple, optional): Fixed range for Z-axis (min, max).
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    seq_len, hidden_dim = activations.shape
    
    # Calculate RMS
    rms = torch.sqrt(torch.mean(activations.float() ** 2)).item()
    title = f"{title}\nRMS: {rms:.4f}"
    
    # plot line-by-line (token-by-token) along the hidden dimension
    for i in range(seq_len):
        # x is constant = i
        # y is 0 to hidden_dim
        # z is activation values
        
        xs = np.full(hidden_dim, i)
        ys = np.arange(hidden_dim)
        zs = activations[i].cpu().numpy()
        
        ax.plot(xs, ys, zs, color='b', alpha=0.6)

    ax.set_xlabel('Tokens')
    ax.set_ylabel('Channels')
    ax.set_zlabel('Activation')
    ax.set_title(title)
    
    if z_lim:
        ax.set_zlim(z_lim)
    
    # Set x-ticks to tokens
    ax.set_xticks(np.arange(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
