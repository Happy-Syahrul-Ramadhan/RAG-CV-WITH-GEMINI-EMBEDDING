"""
Streamlit Chatbot dengan ChromaDB - PDF Q&A
Chatbot yang menggunakan ChromaDB untuk menyimpan dan query embedding.
Auto-load data saat aplikasi dibuka.
"""

import streamlit as st
from typing import List

from embedding_service import GeminiEmbeddingService, GroqGenerationService
from chromadb_service import ChromaDBService
from config import Config


# Page config
st.set_page_config(
    page_title="Chatbot RAG",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Chatbot RAG")
st.caption("Tanya jawab dengan dokumen PDF menggunakan LLM")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chroma_service" not in st.session_state:
    st.session_state.chroma_service = None
if "embed_service" not in st.session_state:
    st.session_state.embed_service = None
if "gen_service" not in st.session_state:
    st.session_state.gen_service = None
if "collection_loaded" not in st.session_state:
    st.session_state.collection_loaded = False
if "auto_load_attempted" not in st.session_state:
    st.session_state.auto_load_attempted = False

# Auto-load ChromaDB on first run
if not st.session_state.auto_load_attempted:
    st.session_state.auto_load_attempted = True
    
    try:
        with st.spinner("Loading ChromaDB collection..."):
            # Determine mode from config
            use_cloud = Config.USE_CHROMA_CLOUD
            
            # Initialize services
            st.session_state.chroma_service = ChromaDBService(use_cloud=use_cloud)
            st.session_state.embed_service = GeminiEmbeddingService()
            st.session_state.gen_service = GroqGenerationService()
            
            # Get collection
            st.session_state.chroma_service.get_or_create_collection()
            
            # Get document count
            count = st.session_state.chroma_service.count_documents()
            
            if count == 0:
                st.warning("Collection is empty. Please store embeddings first using store_to_chromadb.py")
            else:
                st.session_state.collection_loaded = True
                mode = "Cloud" if use_cloud else "Local"
                st.success(f"Loaded {count} documents from ChromaDB ({mode})!")
                
    except Exception as e:
        st.error(f"Error loading ChromaDB: {e}")
        st.info("Please check your configuration in .env file")

# Sidebar - Settings
with st.sidebar:
    st.header("Chatbot with RAG")
    
    # Settings
    st.subheader("Query Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)

# Main chat interface
if not st.session_state.collection_loaded:
    st.info("Collection is not loaded or empty. Please store embeddings first.")
    st.code("python store_to_chromadb.py your_file_embedding.json --cloud")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Tanya sesuatu tentang dokumen..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and generating answer..."):
                # 1. Embed the question
                question_embedding = st.session_state.embed_service.embed_text(prompt)
                
                # 2. Query ChromaDB
                results = st.session_state.chroma_service.query_embeddings(
                    query_embeddings=[question_embedding],
                    n_results=top_k
                )
                
                # 3. Extract results
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                if not documents:
                    st.error("No relevant documents found.")
                    st.stop()
                
                # 4. Generate answer with LLM
                source_name = metadatas[0].get('source_file', 'document') if metadatas else 'document'
                answer = st.session_state.gen_service.ask_with_context(
                    question=prompt,
                    context_chunks=documents,
                    source_name=source_name
                )
                
                # 5. Format citations
                citations = []
                for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
                    similarity = (1 - dist) * 100  # Convert distance to similarity percentage
                    
                    citation = f"**[{i}]** Similarity: {similarity:.1f}%"
                    
                    if meta:
                        if 'page_number' in meta:
                            citation += f" | Page: {meta['page_number']}"
                        if 'chunk_index' in meta:
                            citation += f" | Chunk: {meta['chunk_index']}"
                        if 'source_file' in meta:
                            citation += f" | Source: {meta['source_file']}"
                    
                    citations.append(citation)
                
                full_answer = answer + "\n\n---\n**Sources:**\n" + "\n".join(citations)
                
                # Display answer
                st.markdown(full_answer)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer
                })
                
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# Clear chat button in sidebar
with st.sidebar:
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
