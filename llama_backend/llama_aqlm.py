import torch
import torch.nn as nn
from typing import Dict, Any

class LlamaAQLM(nn.Module):
    """
    A wrapper for AQLM-quantized Llama models to be used with custom evaluation scripts.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        print(f"LlamaAQLM: Loading quantized model from {model_path}...")
        # Since the model was saved as a single object, we use torch.load
        # In PyTorch 2.6+, weights_only defaults to True, which fails for full model objects.
        self.model = torch.load(model_path, map_location=device, weights_only=False)
        self.model.eval()
        self.device = torch.device(device)
        self.config = self.model.config

    def single_step(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Performs a single forward pass.
        
        Args:
            inputs: A dictionary containing 'input_ids' and optionally 'attention_mask'.
            
        Returns:
            torch.Tensor: The logits from the model.
        """
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        return outputs.logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, **kwargs):
        """
        Standard generation interface.
        """
        return self.model.generate(input_ids.to(self.device), max_new_tokens=max_new_tokens, **kwargs)
