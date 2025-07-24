# A-MultiHop-Rag-for-Nepali-Constitution


# 🧠 Multi-hop RAG Pipeline using LLaMA-3 + Qdrant + Groq

This project demonstrates a **multi-hop Retrieval-Augmented Generation (RAG)** pipeline using:

- **LLaMA-3 8B via Groq API** (for ultra-fast inference)
- **LangChain** (for chaining and orchestration)
- **Qdrant** (for vector-based retrieval)
- **BAAI bge-small embeddings** (for semantic chunking)
- **Gradio** (for an interactive interface)

---

## 🚀 Features

✅ Multi-hop QA: Decomposes complex questions into multiple sub-questions  
✅ Fast LLM Inference: Powered by Groq’s blazing-fast LLaMA-3 API  
✅ Vector Search: Retrieves context chunks from Qdrant DB  
✅ Modular: Clean, reusable agents (`decomposer`, `retriever`, `synthesizer`)  
✅ GPU/Accelerated: TinyLlama (optional) runs on local GPU  
✅ Gradio Frontend: Simple UI for testing and demos

---

## 📁 Project Structure

```bash
.
├── rag/
│   ├── decomposer.py       # Breaks down complex queries
│   ├── retriever.py        # Uses HuggingFace + Qdrant for search
│   ├── synthesizer.py      # Synthesizes final answer from sub-answers
│   ├── llm_provider.py     # Loads LLaMA-3 via Groq API
│   ├── ingest.py           # Loads your documents into Qdrant
│   └── multi_hop_agent.py  # Full agent logic (decompose → retrieve → synthesize)
├── interface/
│   └── app.py              # Gradio app
├── alltest.py              # Testing script for each agent
└── README.md               # You're here
