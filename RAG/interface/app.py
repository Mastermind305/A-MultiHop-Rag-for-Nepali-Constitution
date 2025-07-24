

import gradio as gr
from rag.multi_hop_agent import multi_hop_answer

iface = gr.Interface(
    fn=multi_hop_answer,
    inputs="text",
    outputs="text",
    title="Legal RAG with Groq LLaMA",
    description="Ask a legal question. The system will decompose, retrieve, and synthesize an answer using Groq-hosted LLaMA 3."
)

if __name__ == "__main__":
    iface.launch()
