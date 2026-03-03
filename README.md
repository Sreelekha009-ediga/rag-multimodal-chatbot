# RAG-based Multimodal Chatbot

**Retrieval-Augmented Generation chatbot with PDF ingestion, document summarization, and grounded answers using local LLMs.**

Live Demo (local): Run `streamlit run app.py`  
GitHub: https://github.com/YOUR-USERNAME/rag-multimodal-chatbot

### Key Highlights
- Built a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions using external knowledge sources.
- Enabled PDF ingestion, document summarization, and image understanding for multimodal interaction.
- Implemented document chunking (512 tokens), embedding generation (BGE-small-en-v1.5), semantic retrieval, and response synthesis using Llama 3.1 (Ollama).
- Improved response accuracy and factual grounding by combining retrieval with generative models + reranking.
- Tech Stack: Python, RAG Architecture, LlamaIndex, ChromaDB, HuggingFace Embeddings, Ollama (Llama 3.1 8B), Streamlit.

### Project Structure
├── app.py                # Streamlit chat UI
├── src/
│   ├── ingest.py         # PDF loading → chunk → embed → store in Chroma
│   ├── rag.py            # Query engine + reranking + response
│   └── utils.py          
├── data/                 
├── chroma_db/            
├── requirements.txt
└── README.md


### How to Run Locally

1. **Install Ollama** & pull model:
ollama pull llama3.1:8b
ollama serve                # keep this terminal running


2. **Install Python dependencies**:
pip install -r requirements.txt


3. **Add PDFs**:
- Put 1–2 PDF files in `data/` folder

4. **Index documents** (run once):

python src/ingest.py


5. **Launch the chatbot**:
streamlit run app.py