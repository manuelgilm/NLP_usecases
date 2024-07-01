from typing import Optional

from langchain_openai import ChatOpenAI


def get_chat_model(temperature: Optional[float] = 0.5):
    """
    Get the chat model.

    :param model: The model to use.
    :param temperature: The temperature to use.
    """
    chat_model = ChatOpenAI(temperature=temperature)
    return chat_model
