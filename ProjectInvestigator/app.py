import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ollama Web UI Clone",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR OLLAMA UI CLONE ---
st.markdown("""
<style>
    /* Dark theme mimicking Ollama Web UI */
    :root {
        --background-color: #1a1b1e;
        --sidebar-color: #101114;
        --text-color: #ececec;
        --card-bg: #25262b;
        --card-border: #373a40;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Central Hero Text */
    .hero-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        margin-top: 5vh;
        margin-bottom: 2rem;
        color: #FFFFFF;
    }
    
    .hero-icon {
        text-align: center;
        font-size: 4rem;
        margin-top: 10vh;
        margin-bottom: 1rem;
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar-menu-item {
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 5px;
        cursor: pointer;
        display: flex;
        align-items: center;
        color: #c1c2c5;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .sidebar-menu-item:hover {
        background-color: #2c2e33;
        color: white;
    }
    .sidebar-icon {
        margin-right: 15px;
        font-size: 1.1rem;
    }
    
    /* Custom buttons (for the 2x2 grid) */
    .prompt-card {
        background-color: transparent;
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .prompt-card:hover {
        background-color: var(--card-bg);
    }
    .prompt-card-title {
        font-weight: 600;
        color: #e4e5e7;
        font-size: 0.95rem;
        margin-bottom: 5px;
    }
    .prompt-card-desc {
        color: #8c8f96;
        font-size: 0.85rem;
    }
    
    /* Top nav drop down fake styling */
    .stSelectbox label {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND MODEL & FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    return uploaded_file.read().decode('utf-8')

def chunk_text(text, window_size, overlap):
    words = text.split()
    chunks = []
    if window_size <= overlap: return []
    step = window_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + window_size])
        if chunk.strip(): chunks.append(chunk)
    return chunks

@st.cache_data(show_spinner="Processing Document...")
def process_document(_uploaded_file, window_size, overlap):
    _uploaded_file.seek(0)
    raw_text = extract_text_from_file(_uploaded_file)
    chunks = chunk_text(raw_text, window_size, overlap)
    if not chunks: return [], None
    embeddings = model.encode(chunks)
    return chunks, embeddings

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-menu-item"><span class="sidebar-icon">👽</span> New Chat <span style="margin-left:auto;">📝</span></div>
        <div class="sidebar-menu-item"><span class="sidebar-icon">⚙️</span> Modelfiles</div>
        <div class="sidebar-menu-item"><span class="sidebar-icon">✏️</span> Prompts</div>
        <div class="sidebar-menu-item"><span class="sidebar-icon">📄</span> Documents</div>
        <div class="sidebar-menu-item"><span class="sidebar-icon">🔍</span> Search</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
# --- TOP NAV & UPLOAD ---
col_head, col_upload = st.columns([1, 4])
with col_head:
    st.selectbox("Model", ["Project Investigator Engine v1"], label_visibility="collapsed")
with col_upload:
    input_source = st.file_uploader("Upload Knowledge Base", type=["pdf", "txt"], label_visibility="collapsed")

# Hardcoded processing parameters
window_size = 512
overlap = 70

# Process data
chunks_placeholder = []
embeddings = None
if input_source is not None:
    chunks, embeddings = process_document(input_source, window_size, overlap)
    if chunks:
        chunks_placeholder = chunks

# --- INITIALIZE CHAT STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MAIN CONTENT AREA ---
if len(st.session_state.messages) == 0:
    # Show Hero Area if no messages
    st.markdown('<div class="hero-icon">🦙</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">How can I help you today?</div>', unsafe_allow_html=True)
    
    # Display the grid, centered using empty columns
    c_left, c_middle1, c_middle2, c_right = st.columns([1, 2, 2, 1])
    with c_middle1:
        st.markdown("""
        <div class="prompt-card">
            <div class="prompt-card-title">Extract key figures</div>
            <div class="prompt-card-desc">of the main characters or data points</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="prompt-card">
            <div class="prompt-card-title">Help me study</div>
            <div class="prompt-card-desc">create flashcards from the text</div>
        </div>
        """, unsafe_allow_html=True)
    with c_middle2:
        st.markdown("""
        <div class="prompt-card">
            <div class="prompt-card-title">Show me a quotation</div>
            <div class="prompt-card-desc">related to the primary subject matter</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="prompt-card">
            <div class="prompt-card-title">Give me ideas</div>
            <div class="prompt-card-desc">for what to do with the uploaded text</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Send a message"):
    # Render user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response
    with st.chat_message("assistant"):
        if embeddings is not None and len(embeddings) > 0:
            with st.spinner("Processing..."):
                query_embed = model.encode([prompt])
                similarities = cosine_similarity(query_embed, embeddings)[0]
                
                top_k = min(1, len(chunks_placeholder))
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                if len(top_indices) > 0:
                    idx = top_indices[0]
                    response = f"**Top Match** (Confidence: `{similarities[idx]:.2f}`)\n\n{chunks_placeholder[idx]}\n"
                else:
                    response = "No matching context found."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "Ready to assist! Please attach a PDF or TXT to the *File Ingestion* menu in the sidebar so I have context."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
