from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Configuration
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Changed to MPNet
CHUNK_SIZE = 600  # Increased for better context (MPNet handles longer chunks well)
CHUNK_OVERLAP = 100  # Increased overlap for better continuity

def load_pdf_files(data_path: str):
    """Load PDF files with progress tracking"""
    try:
        loader = DirectoryLoader(
            data_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
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
            'normalize_embeddings': True,  # Crucial for MPNet
            'batch_size': 32  # Optimal for this model
        }
    )

def create_vector_store():
    """End-to-end pipeline"""
    # 1. Load documents
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory not found at {DATA_PATH}")
    
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        raise ValueError("No documents loaded")
    
    # 2. Create chunks
    chunks = create_chunks(documents)
    print(f"✓ Created {len(chunks)} chunks (avg. {CHUNK_SIZE} chars)")
    
    # 3. Initialize MPNet embeddings
    embeddings = get_mpnet_embeddings()
    
    # 4. Create and save FAISS store
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"✓ Saved vector store to {DB_FAISS_PATH}")
    return db

if __name__ == "__main__":
    create_vector_store()