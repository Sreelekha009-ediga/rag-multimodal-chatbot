import streamlit as st
from src.ingest import load_or_create_index
from src.rag import create_query_engine, query_rag
import os

st.set_page_config(page_title="RAG Multimodal Chatbot", layout="wide")
st.title("RAG Chatbot – Ask your PDFs!")

# Load index once
@st.cache_resource
def get_query_engine():
    index = load_or_create_index("data")
    return create_query_engine(index)

query_engine = get_query_engine()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.write(f"**File:** {src['file']} | **Page:** {src['page']}")
                    st.caption(src['text'])

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = query_rag(query_engine, prompt)

        st.markdown(response_data["answer"])

        if response_data["sources"]:
            with st.expander("Sources"):
                for src in response_data["sources"]:
                    st.write(f"**File:** {src['file']} | **Page:** {src['page']}")
                    st.caption(src['text'])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["answer"],
        "sources": response_data["sources"]
    })