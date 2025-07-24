# from langchain_community.vectorstores import Qdrant

# from langchain_huggingface import HuggingFaceEmbeddings
# from qdrant_client import QdrantClient
# import os

# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# COLLECTION_NAME = "knowledge_base"

# def get_retriever():
#     client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

#     vectorstore = Qdrant(
#         client=client,
#         collection_name=COLLECTION_NAME,
#         embeddings=embeddings,
#     )
#     return vectorstore.as_retriever(search_kwargs={"k": 5})


from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import os

# Read Qdrant URL and collection name from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional (used only for Qdrant Cloud)
COLLECTION_NAME = "knowledge_base"

# Fail early if QDRANT_URL is not set
if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL environment variable is not set.")

def get_retriever():
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Use consistent BAAI embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Connect LangChain to Qdrant
    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    # Return retriever object with top 5 results
    return vectorstore.as_retriever(search_kwargs={"k": 2})
