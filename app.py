import streamlit as st
import os
from dotenv import load_dotenv

from llm_grok import call_grok
from pdf_loader import load_pdf
from chunker import create_chunks
from embeddings import get_embeddings
from pinecone_db import create_index, get_index, upsert_vectors, clear_index

# Load env variables FIRST
load_dotenv()

st.set_page_config(page_title="PDF RAG System", layout="wide")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY environment variable not set!")
    st.info("Set it with: `export PINECONE_API_KEY=your_key` (Linux/Mac) or `$env:PINECONE_API_KEY=your_key` (Windows PowerShell)")
    st.stop()

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY environment variable not set!")
    st.info("Set it with: `export GROQ_API_KEY=your_key` (Linux/Mac) or `$env:GROQ_API_KEY=your_key` (Windows PowerShell)")
    st.stop()


with st.sidebar:
    st.title("⚙️ Controls")

    chunk_size = st.slider("Chunk Size", 100, 2000, 500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, step=10)

    # Top-K Retrieval Slider
    top_k = st.slider("Top-K Retrieval", 1, 10, 3, step=1,
                      help="How many chunks to retrieve from Pinecone")

    # Confidence Score Threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05,
                                     help="Minimum similarity score for chunks to be considered relevant")

    st.markdown("---")

    # Multiple File Upload with better display
    st.subheader("📎 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    # Display uploaded files nicely
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for i, f in enumerate(uploaded_files, 1):
            st.markdown(f"📄 **{i}.** `{f.name}`")
    else:
        st.info("ℹ️ No files uploaded yet")

    st.markdown("---")

    # Clear Index Button
    if st.button("🗑️ Clear Pinecone Index", type="secondary"):
        with st.spinner("Clearing index..."):
            try:
                clear_index()
                st.session_state.index = None
                st.session_state.embedding_model = None
                st.session_state.last_uploaded_files = []
                st.success("✅ Index cleared! Upload new PDFs.")
            except Exception as e:
                st.error(f"❌ Error clearing index: {e}")

    st.markdown("---")
    st.info("📄 RAG System — Pinecone + Groq + HuggingFace")

# MAIN TITLE
st.title("📄 AI PDF RAG Assistant")

# Initialize session state keys safely
if "index" not in st.session_state:
    st.session_state.index = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

# -----------------------
# PROCESS PDFs
# -----------------------
if uploaded_files:

    # Get current file names
    current_file_names = [f.name for f in uploaded_files]

    # Reset state when different files are uploaded
    if st.session_state.last_uploaded_files != current_file_names:
        st.session_state.index = None
        st.session_state.embedding_model = None
        st.session_state.last_uploaded_files = []

    if st.button("🚀 Process PDFs"):

        with st.spinner("Processing PDFs..."):

            all_documents = []

            # Process each PDF
            for uploaded_file in uploaded_files:
                st.write(f"📄 Loading {uploaded_file.name}...")
                documents = load_pdf(uploaded_file)

                # Add source filename to metadata
                for doc in documents:
                    if hasattr(doc, 'metadata'):
                        doc.metadata["source"] = uploaded_file.name
                    else:
                        doc.metadata = {"source": uploaded_file.name}

                all_documents.extend(documents)

            if not all_documents:
                st.error("❌ No text found in any PDF. Please try other files.")
                st.stop()

            # 2. Split into chunks
            chunks = create_chunks(all_documents, chunk_size, chunk_overlap)
            chunks_text = [doc.page_content for doc in chunks]

            if not chunks_text:
                st.error("❌ No chunks created. Please try other files.")
                st.stop()

            st.info(f"📄 Total chunks from {len(uploaded_files)} file(s): {len(chunks_text)}")

            # 3. Load embedding model (cached)
            @st.cache_resource
            def load_embedding_model():
                return get_embeddings()

            embedding_model = load_embedding_model()

            # 4. Embed all chunks
            with st.spinner("Generating embeddings... (this may take a moment)"):
                chunk_embeddings = embedding_model.embed_documents(chunks_text)

            # 5. Create Pinecone index and upsert
            with st.spinner("Connecting to Pinecone..."):
                create_index()
                index = get_index()

            if index is None:
                st.error("❌ Could not connect to Pinecone. Check your API key and network.")
                st.stop()

            with st.spinner("Uploading vectors to Pinecone..."):
                success = upsert_vectors(index, chunks_text, chunk_embeddings, 
                                        sources=[doc.metadata.get("source", "unknown") for doc in chunks])

            if not success:
                st.error("❌ Failed to upload vectors to Pinecone.")
                st.stop()

            # 6. Save to session state
            st.session_state.index = index
            st.session_state.embedding_model = embedding_model
            st.session_state.last_uploaded_files = current_file_names

            st.success(f"✅ {len(uploaded_files)} PDF(s) Processed: {len(chunks_text)} chunks stored in Pinecone.")

# CHAT SECTION
if st.session_state.index is not None:

    st.markdown("---")
    st.markdown("## 💬 Ask Questions from Your PDFs")

    user_question = st.text_input("Type your question here...", key="question_input")

    ask_button = st.button("Ask ➤")

    if ask_button and user_question.strip():

        with st.spinner("Thinking... 🤔"):

            index = st.session_state.index
            embedding_model = st.session_state.embedding_model

            # Embed the user query
            query_vector = embedding_model.embed_query(user_question)

            # Search Pinecone with top_k from slider
            try:
                results = index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
            except Exception as e:
                st.error(f"❌ Pinecone query failed: {e}")
                st.stop()

            # Extract matching chunks with scores
            matched_chunks = []
            chunk_scores = []
            chunk_sources = []

            if hasattr(results, 'matches'):
                # Object syntax (Pinecone v5+)
                for match in results.matches:
                    score = match.score if hasattr(match, 'score') else 0.0
                    if score >= confidence_threshold and match.metadata and "text" in match.metadata:
                        matched_chunks.append(match.metadata["text"])
                        chunk_scores.append(score)
                        chunk_sources.append(match.metadata.get("source", "unknown"))
            elif isinstance(results, dict) and "matches" in results:
                # Dict syntax
                for match in results["matches"]:
                    score = match.get("score", 0.0)
                    if score >= confidence_threshold and match.get("metadata") and "text" in match.get("metadata", {}):
                        matched_chunks.append(match["metadata"]["text"])
                        chunk_scores.append(score)
                        chunk_sources.append(match["metadata"].get("source", "unknown"))

            if not matched_chunks:
                st.warning("⚠️ No relevant context found above the confidence threshold. Try lowering the threshold or rephrasing your question.")
                st.stop()

            context = "\n\n".join(matched_chunks)

            # Show retrieved context with confidence scores
            st.markdown("### 📌 Retrieved Context")
            with st.expander(f"View {len(matched_chunks)} Retrieved Chunks (Top-K: {top_k})"):
                for i, (chunk, score, source) in enumerate(zip(matched_chunks, chunk_scores, chunk_sources)):
                    st.markdown(f"**Chunk {i+1}** | Source: `{source}` | Score: `{score:.4f}`")
                    st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.markdown("---")

            # Build prompt
            prompt = f"""You are an AI assistant. Answer only from the given context. 
If the answer is not in the context, say: "I could not find this in the document."

Context:
{context}

Question:
{user_question}
"""
            # Call Groq API
            try:
                with st.spinner("Getting answer from AI..."):
                    answer = call_grok(prompt)
            except Exception as e:
                st.error(f"❌ API error: {e}")
                st.stop()

            st.markdown("### 🤖 AI Answer")
            st.success(answer)

            # Show average confidence
            if chunk_scores:
                avg_score = sum(chunk_scores) / len(chunk_scores)
                st.caption(f"📊 Average Retrieval Confidence: `{avg_score:.4f}` | Chunks Used: {len(matched_chunks)}/{top_k}")

    elif ask_button and not user_question.strip():
        st.warning("Please enter a question first.")