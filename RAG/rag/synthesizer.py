
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rag.llm_provider import get_llm

def get_synthesizer():
    llm = get_llm()
    prompt = PromptTemplate.from_template("""
Given the following sub-answers from a legal knowledge base, synthesize a clear and concise final answer.

{sub_answers}

Final Answer:
""")
    return LLMChain(llm=llm, prompt=prompt, output_key="text")
