import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page config for wider layout
st.set_page_config(page_title="Project Investigator", layout="wide")

st.title("Project Investigator 🕵️‍♂️")
st.write("Upload a PDF to analyze and search its contents using AI embeddings and Cosine Similarity.")

# --- Caching the model to avoid reloading on every interaction ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    # PyMuPDF can open streams or byte buffers
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text() + "\n"
    return text

def chunk_text(text, chunk_size, overlap):
    # Standard character/word sliding window chunking
    # Here we'll do word level.
    words = text.split()
    chunks = []
    if chunk_size <= overlap:
        st.error("Chunk size must be greater than overlap.")
        return []

    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Cache the extraction and embedding process based on file and parameters
@st.cache_data
def process_pdf(_uploaded_file, chunk_size, overlap):
    _uploaded_file.seek(0)
    raw_text = extract_text_from_pdf(_uploaded_file)
    chunks = chunk_text(raw_text, chunk_size, overlap)
    
    # Generate embeddings
    if not chunks:
        return [], None
        
    embeddings = model.encode(chunks)
    return chunks, embeddings

# --- UI Configuration ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    st.subheader("Chunking Parameters")
    chunk_size = st.slider("Chunk Size (Words)", min_value=50, max_value=1000, value=200, step=50)
    overlap = st.slider("Overlap (Words)", min_value=10, max_value=500, value=50, step=10)

# --- Main App Logic ---
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        chunks, embeddings = process_pdf(uploaded_file, chunk_size, overlap)
        
    if chunks:
        st.success(f"Document processed successfully! Extracted {len(chunks)} chunks.")
        
        st.divider()
        st.subheader("Search the Document")
        query = st.text_input("Ask a question about the document:")
        top_k = st.slider("Number of results to return", min_value=1, max_value=10, value=3)
        
        if query:
            query_embedding = model.encode([query])
            
            # Calculate Cosine Similarity
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Get top K indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            st.write("---")
            st.write("### Most Relevant Sections (Cosine Similarity)")
            for i, idx in enumerate(top_indices):
                with st.expander(f"Result {i+1} - Similarity Score: {similarities[idx]:.4f}", expanded=True):
                    st.write(chunks[idx])
else:
    st.info("Please upload a PDF from the sidebar to begin.")
