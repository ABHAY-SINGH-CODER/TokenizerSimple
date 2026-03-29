# Project Investigator

A local, offline, lightweight Retrieval-Augmented Generation (RAG) implementation designed to ingest PDF/TXT documents and allow conversational querying. The application features a clean, minimalist UI purposefully built to clone the sleek aesthetic of the Ollama Web UI.

## Features
- **Complete Document Ingestion**: Upload PDFs and raw Text files directly into the web interface.
- **Intelligent Chunking Pipeline**: Documents are mapped using a Sliding Window approach (Window Size: 512, Overlap: 70) for context preservation.
- **Advanced Local Embeddings**: Built on `sentence-transformers/all-MiniLM-L6-v2` to run lightning fast without internet API requirements.
- **Precision Retrieval Engine**: Cosine Similarity matching guarantees maximum accuracy in finding the exact chunk related to your prompt.
- **Ollama Web UI Styling**: Enjoy a ChatGPT-style Chat Interface paired with modern dark mode styling, hover cards, and robust side navigation.

## Installation

1. **Clone the repository and enter the directory**
   ```bash
   cd TokenizerSimple/ProjectInvestigator
   ```

2. **Activate your Python virtual environment**
   ```bash
   # MacOS/Linux
   source ../venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage
Simply start the Streamlit application and open `http://localhost:8501`. 
1. Look to the top left area ("Model Selection" zone) and upload your `.pdf` or `.txt` file.
2. The neural relay will instantly chunk and embed the entire document.
3. Once completed, begin querying your document using the chat bar at the bottom! The investigator will output the single most contextually relevant snippet matching your request.

## Architecture & Workflow Structure
To read exactly how the math, algorithms, and logic pipeline functions under the hood (step-by-step from Ingestion to Embedding to Output), check the included `workflow.txt` file!
