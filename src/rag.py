from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))

llm = Ollama(
    model=os.getenv("OLLAMA_MODEL"),
    request_timeout=600.0,   # ← change to 600 seconds (10 min)
    temperature=0.7,
)
def create_query_engine(index: VectorStoreIndex):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # Optional reranker (makes results better)
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=5,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[reranker],
        llm=llm,
        response_mode="compact",
    )

    return query_engine

def query_rag(query_engine, question: str):
    response = query_engine.query(question)
    return {
        "answer": str(response),
        "sources": [
            {
                "text": node.node.text[:300] + "...",
                "file": node.node.metadata.get("file_name", "Unknown"),
                "page": node.node.metadata.get("page_label", "N/A")
            }
            for node in response.source_nodes[:3]
        ]
    }