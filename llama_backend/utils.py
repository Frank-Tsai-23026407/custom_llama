from transformers import StoppingCriteria
import torch 

def input_formatting(history, input_text):
    """Formats conversation history and new input for a chat model.

    This function constructs a prompt string by concatenating the history of
    user and assistant messages, followed by the new user input. The format
    uses specific tokens like `<|user|>:`, `<|assistant|>:`, and `</s>` to
    delimit turns, making it suitable for instruction-tuned models.

    Args:
        history (list of list of str): A list of conversation turns, where each
            turn is a `[user_message, assistant_message]` pair.
        input_text (str): The latest user message to be added to the prompt.

    Returns:
        str: A fully formatted string ready to be tokenized by the model.
    """
    history_transformer_format = history + [[input_text, ""]]
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    return messages

class StopOnTokens(StoppingCriteria):
    """A stopping criterion that halts generation upon encountering a specific token.

    This class provides a simple way to stop text generation when the model
    emits a designated end-of-sequence (EOS) token. It is designed to be used
    with the `transformers` library's generation utilities.

    Note:
        The token ID `2` is specific to the TinyLlama tokenizer and corresponds
        to the EOS token.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Checks if the last generated token is the EOS token.

        Args:
            input_ids (torch.LongTensor): A tensor of shape (batch_size, sequence_length)
                containing the token IDs generated so far.
            scores (torch.FloatTensor): The logits for the last generated token.
            **kwargs: Additional keyword arguments passed by the generation utility.

        Returns:
            bool: `True` if the last token is the EOS token, `False` otherwise.
        """
        return input_ids[0][-1] == 2 # EOS token id for TinyLlama
