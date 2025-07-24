from langchain.chat_models import ChatOpenAI
import os

def get_llm():
    return ChatOpenAI(
        model="llama3-8b-8192",  # Groq LLaMA-3.1 8B
        openai_api_key="Your Groq Api Key",
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.3
    )
