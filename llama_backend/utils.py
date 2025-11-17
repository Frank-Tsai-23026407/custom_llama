from transformers import StoppingCriteria
import torch 

def input_formatting(history, input_text):
    """
    Formats the input history and new text into a single string for the model.

    This function takes a history of conversation turns and a new input text, and formats
    them into a single string with specific user and assistant tags, suitable for the model.

    Args:
        history (list): A list of previous conversation turns, where each turn is a list
            of two strings: [user_message, assistant_message].
        input_text (str): The new input text from the user.

    Returns:
        str: A formatted string containing the entire conversation history and the new input.
    """
    history_transformer_format = history + [[input_text, ""]]
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    return messages

class StopOnTokens(StoppingCriteria):
    """
    A stopping criteria that stops generation when a specific token is generated.

    This class implements the `StoppingCriteria` interface from the `transformers` library.
    It is used to stop the text generation process when the last generated token is the
    end-of-sequence (EOS) token.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether the generation should be stopped.

        Args:
            input_ids (torch.LongTensor): The input token IDs.
            scores (torch.FloatTensor): The scores of the last generated tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the generation should be stopped, False otherwise.
        """
        return input_ids[0][-1] == 2 # EOS token id for TinyLlama
