import os
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

# Force local models – this fixes the OpenAI default error
Settings.embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
Settings.llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), request_timeout=300.0)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Global index
index = None

def load_or_create_index(data_dir: str = "data"):
    global index

    if index is not None:
        return index

    print("Current working directory:", os.getcwd())
    print("Data directory full path:", os.path.abspath(data_dir))
    print("Files in data dir:", os.listdir(data_dir) if os.path.exists(data_dir) else "Directory not found!")

    print("Initializing Chroma client...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("Getting/creating collection 'echoflow_rag'...")
    chroma_collection = chroma_client.get_or_create_collection("echoflow_rag")
    print("Collection count before indexing:", chroma_collection.count())

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Loading documents from:", data_dir)
    documents = SimpleDirectoryReader(
        data_dir,
        recursive=True,
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()
    print("Loaded", len(documents), "documents")

    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    if chroma_collection.count() == 0:
        print("Creating new index...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
            transformations=[node_parser],
            show_progress=True
        )
        print("New index created! Nodes added.")
    else:
        print("Loading existing index...")
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    print("Final collection count:", chroma_collection.count())
    return index