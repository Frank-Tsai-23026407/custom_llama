# TinyLlama Playground

This playground provides an interactive web interface to experiment with TinyLlama and other compatible models. It supports dynamic model loading, multiple generation backends, and an extensible registry system for adding custom generation logic.

## 🚀 Quick Start

### 1. Environment Setup

Ensure you have the `llama-env` Conda environment set up. If not, create it and install dependencies:

```bash
conda create -n llama-env python=3.10
conda activate llama-env
pip install -r playground/requirements.txt
```

### 2. Run the App

Activate the environment and launch the app:

```bash
conda activate llama-env
python playground/app.py
```

The app will start a local server, typically at `http://127.0.0.1:7860`. Open this URL in your browser.

## ✨ Features

-   **Dynamic Model Loading**: Switch models on the fly by changing the "Model Path".
-   **Multiple Backends**:
    -   **Hugging Face**: Uses the standard `transformers` library.
    -   **Llama Custom**: Uses our custom `LlamaMyModel` implementation.
    -   **Random Word**: A demo backend that generates random words.
-   **Extensible Registry**: Easily add your own generation functions using a simple decorator in separate files.

## 🛠️ How to Extend

You can register new generation strategies in separate files without modifying the core app logic.

### Adding a New Generation Function

1.  Create a new Python file in the `playground` directory (e.g., `my_experiment.py`).
2.  Import `register_generation_function` from `utils`.
3.  Define your function and decorate it.

**Example (`playground/my_experiment.py`):**

```python
from utils import register_generation_function

@register_generation_function("My Experiment")
def my_custom_generate(model_path, history_transformer_format):
    yield "Running experiment..."
    # Your custom logic here
    yield "Experiment complete!"
```

4.  **Important**: Import your new file in `playground/app.py` so it gets registered:

```python
# playground/app.py
import my_experiment 
```

The new option "My Experiment" will automatically appear in the "Backend" dropdown.

## 🧩 Architecture

-   **`app.py`**: Main Gradio application.
-   **`utils.py`**: Contains the registry logic (`GENERATION_FUNCTIONS`, `@register_generation_function`) and model loading helper (`get_or_load_model`).
-   **`custom_functions.py`**: Example of a separate file defining a custom generation function ("Random Word").
-   **`llama_backend/llama_my.py`**: Contains the `LlamaMyModel` class.

## ❓ Troubleshooting

-   **ModuleNotFoundError**: Make sure you are in the `llama-env` environment.
-   **Backend Not Showing**: Ensure you have imported your custom file in `app.py`.
