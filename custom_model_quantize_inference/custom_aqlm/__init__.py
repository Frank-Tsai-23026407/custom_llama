# Expose QuantizedLinear to satisfy transformers when local aqlm shadows the official library
from .utils.aq import QuantizedLinear
