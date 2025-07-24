# A-MultiHop-RAG-for-Nepali-Constitution

A multi-hop Retrieval-Augmented Generation (RAG) pipeline for querying the Nepali Constitution, leveraging **LLaMA-3 8B** via Groq API, **LangChain**, **Qdrant**, and **Gradio**.

## 🚀 Features

- **Multi-hop QA**: Decomposes complex questions into sub-questions for precise answers.
- **Fast Inference**: Utilizes Groq’s LLaMA-3 API for ultra-fast LLM performance.
- **Vector Search**: Retrieves relevant context chunks using Qdrant and **BAAI bge-small** embeddings.
- **Modular Design**: Reusable agents for decomposition, retrieval, and synthesis.
- **GPU Support**: Optional local TinyLlama for GPU-accelerated environments.
- **Interactive UI**: Gradio-based frontend for easy testing and demos.

## 📁 Project Structure

```
├── rag/
│   ├── decomposer.py       # Breaks down complex queries
│   ├── retriever.py        # Handles vector search with HuggingFace + Qdrant
│   ├── synthesizer.py      # Combines sub-answers into a final response
│   ├── llm_provider.py     # Integrates LLaMA-3 via Groq API
│   ├── ingest.py           # Loads documents into Qdrant
│   └── multi_hop_agent.py  # Orchestrates the full multi-hop pipeline
├── interface/
│   └── app.py              # Gradio-based interactive UI
└── README.md               # Project documentation
```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/A-MultiHop-RAG-for-Nepali-Constitution.git](https://github.com/Mastermind305/A-MultiHop-Rag-for-Nepali-Constitution.git
   cd A-MultiHop-RAG-for-Nepali-Constitution
   ```

2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. Save the following as `requirements.txt`:
   ```
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
   ```

## 🛠️ TODO

- ✅ Add GPU support for local TinyLlama
- ✅ Replace OpenAI with Groq API
- ✅ Implement sub-question limit
- ⏳ Support Mixtral or Gemma via Groq
- ⏳ Add streaming output to Gradio UI

## 📬 Contact

For questions, feedback, or collaboration, reach out to:  
**Email**: [gautamaayush305@gmail.com](mailto:gautamaayush305@gmail.com)
