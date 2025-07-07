
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

import os

# Configuration
DATA_PATH = "./data/"
CHROMA_DB_PATH = "vectorstore/chroma"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def load_pdf_files(data_path: str):
    """Load PDF files using PyMuPDFLoader with better accuracy"""
    try:
        loader = DirectoryLoader(
            data_path,
            glob="*.pdf",
            loader_cls=PyMuPDFLoader,  # 🔄 Use PyMuPDFLoader instead of PyPDFLoader
            show_progress=True
        )
        docs = loader.load()
        print(f"✓ Loaded {len(docs)} pages from {len(list(os.listdir(data_path)))} PDFs")
        return docs
    except Exception as e:
        print(f"✗ Failed to load PDFs: {str(e)}")
        return []

def create_chunks(documents):
    """Optimized chunking for MPNet embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def get_mpnet_embeddings():
    """Initialize MPNet with optimal settings"""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )

def create_chroma_vector_store():
    """End-to-end Chroma pipeline"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory not found at {DATA_PATH}")
    
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        raise ValueError("No documents loaded")

    # Optionally filter/clean here

    chunks = create_chunks(documents)
    print(f"✓ Created {len(chunks)} chunks (avg. {CHUNK_SIZE} chars)")

    embeddings = get_mpnet_embeddings()

    # Persist directory for Chroma
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    db.persist()
    print(f"✓ Saved Chroma vector store to {CHROMA_DB_PATH}")
    return db

if __name__ == "__main__":
    create_chroma_vector_store()
