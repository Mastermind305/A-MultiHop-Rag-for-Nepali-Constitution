


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1️⃣ Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = "knowledge_base"
PDF_PATH = r"data/Constitution-of-Nepal_2072_Eng_www.moljpa.gov_.npDate-72_11_16.pdf"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("❌ QDRANT_API_KEY or QDRANT_URL not set in environment variables!")

# 2️⃣ Initialize Qdrant Client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 3️⃣ Create collection (if not exists)
if not client.collection_exists(COLLECTION_NAME):
    print(f"🛠️ Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
else:
    print(f"✅ Collection '{COLLECTION_NAME}' already exists.")

# 4️⃣ Load and annotate PDF
print("📖 Loading Constitution PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Add page-level metadata
for i, doc in enumerate(documents):
    doc.metadata["source"] = "Nepal Constitution"
    doc.metadata["page_number"] = i + 1

# 5️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Add chunk-level metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i + 1

print(f"📝 Split into {len(chunks)} chunks.")

# 6️⃣ Embed
print(f"🔬 Using embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 7️⃣ Upload to Qdrant
print(f"🚀 Uploading chunks to collection '{COLLECTION_NAME}'...")
qdrant_store = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

qdrant_store.add_documents(chunks)
print("✅ Ingestion complete with metadata!")


