# A-MultiHop-Rag-for-Nepali-Constitution  
🧠 Multi-hop RAG Pipeline using LLaMA-3 + Qdrant + Groq  

This project demonstrates a multi-hop Retrieval-Augmented Generation (RAG) pipeline using:

- **LLaMA-3 8B** via Groq API (for ultra-fast inference)  
- **LangChain** (for chaining and orchestration)  
- **Qdrant** (for vector-based retrieval)  
- **BAAI bge-small** embeddings (for semantic chunking)  
- **Gradio** (for an interactive interface)  

---

## 🚀 Features

✅ Multi-hop QA: Decomposes complex questions into multiple sub-questions  
✅ Fast LLM Inference: Powered by Groq’s blazing-fast LLaMA-3 API  
✅ Vector Search: Retrieves context chunks from Qdrant DB  
✅ Modular: Clean, reusable agents (decomposer, retriever, synthesizer)  
✅ GPU/Accelerated: TinyLlama (optional) runs on local GPU  
✅ Gradio Frontend: Simple UI for testing and demos  

---

## 📁 Project Structure

├── rag/
│ ├── decomposer.py # Breaks down complex queries
│ ├── retriever.py # Uses HuggingFace + Qdrant for search
│ ├── synthesizer.py # Synthesizes final answer from sub-answers
│ ├── llm_provider.py # Loads LLaMA-3 via Groq API
│ ├── ingest.py # Loads your documents into Qdrant
│ └── multi_hop_agent.py # Full agent logic (decompose → retrieve → synthesize)
├── interface/
│ └── app.py # Gradio app
└── README.md # You're here


---

## 📦 Requirements

Save the following as `requirements.txt`:

langchain>=0.2.0
langchain-community>=0.0.27
langchain-core>=0.1.50
langchain-openai>=0.1.6
langchain-huggingface>=0.0.3
langchain-qdrant>=0.1.2

openai>=1.30.1
transformers>=4.41.1
accelerate>=0.30.1
torch>=2.2.2
qdrant-client>=1.9.0
gradio>=4.44.1
python-dotenv>=1.0.1


Install dependencies with:

```bash
pip install -r requirements.txt

---
## 🛠️ TODO

- ✅ Add GPU support for local TinyLlama  
- ✅ Replace OpenAI with Groq  
- ✅ Add sub-question limit  
- ⏳ Support Mixtral or Gemma (via Groq)  
- ⏳ Add streaming output to Gradio  

---

## 📬 Contact

For questions, feedback, or collaboration:  
**Email:** [gautamaayush305@gmail.com](mailto:gautamaayush305@gmail.com)

Feel free to reach out!
