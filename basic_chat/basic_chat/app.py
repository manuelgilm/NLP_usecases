import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

from basic_chat.prompting.prompts import get_chat_model

load_dotenv()

st.title("Basic Chat App")


@st.cache_resource()
def get_chatbot():
    """
    Get the chatbot.
    """
    return get_chat_model()


chat_bot = get_chatbot()

user_input = st.text_input("Enter your message")
ai_message = (
    "You are a nice AI bot that helps a user figure out what to do next."
)
response = chat_bot.invoke(
    [
        SystemMessage(content=ai_message),
        HumanMessage(content=user_input),
    ]
)
st.write(f"Response: {response.content}")
