
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rag.llm_provider import get_llm

def get_decomposer():
    llm = get_llm()
    prompt = PromptTemplate.from_template("""
You are a legal assistant.

Break down the following legal question into at most two simple, answerable sub-questions.

Format your response as:
1. First sub-question
2. Second sub-question (if needed)

Question: {question}

Sub-questions:
""")
    return LLMChain(llm=llm, prompt=prompt, output_key="text")
