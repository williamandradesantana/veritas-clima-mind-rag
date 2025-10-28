import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.pinecone import PineconeVectorStore

from loaders.csv_loader import CSVLoader
from loaders.pdf_loader import PDFLoader

from services.env_loader import EnvLoader
from services.pinecone_service import PineconeService


def gather_documents(data_folder="data"):
    """
    Collects and processes all CSV and PDF documents from the specified folder.
    
    Args:
        data_folder (str): Root folder containing 'spreadsheets' and 'pdfs' subfolders
        
    Returns:
        list: List of Document objects ready for indexing
    """
    documents = []
    
    # Process CSV files
    csv_folder = Path(data_folder) / "spreadsheets"
    for csv_file in csv_folder.glob("*.csv"):
        texts = CSVLoader(csv_file).to_text_list()
        documents.extend(Document(text=t) for t in texts)
        print(f"✅ CSV '{csv_file.name}' processed successfully.")
    
    # Process PDF files
    pdf_folder = Path(data_folder) / "pdfs"
    for pdf_file in pdf_folder.glob("*.pdf"):
        chunks = PDFLoader(pdf_file).extract_text_chunks()
        documents.extend(Document(text=c) for c in chunks)
        print(f"✅ PDF '{pdf_file.name}' processed successfully.")
    
    return documents


def main():
    env = EnvLoader()
    
    pc = PineconeService(api_key=env.PINECONE_API_KEY)
    
    index_name = "llama-integration-example"
    namespace = "default"
    
    # Connect to Pinecone index
    print(f"🔹 Connecting to Pinecone index '{index_name}'...")
    pinecone_index = pc.pc.Index(index_name)
    
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, 
        namespace=namespace
    )
    
    # Initialize embedding model (Ollama local model)
    embedding_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # Create storage context for vector operations
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Check if index already has vectors
    stats = pinecone_index.describe_index_stats()
    total_vectors = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    
    if total_vectors == 0:
        print(f"Index '{index_name}' is empty. Indexing all PDFs and CSVs from '{Path('data')}'...")
        
        # Gather all documents from data folder
        documents = gather_documents("data")
        
        if documents:
            # Create index and embed all documents
            print(f"Embedding and indexing {len(documents)} documents...")
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                embed_model=embedding_model
            )
            print(f"{len(documents)} documents indexed successfully!")
        else:
            print("⚠️ No documents found to index.")
            return
    else:
        print(f"📊 Loading existing index with {total_vectors} vectors...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            embed_model=embedding_model
        )
        print(f"Index loaded successfully with {total_vectors} vectors.")
    
    # Initialize LLM (Large Language Model)
    print("🤖 Initializing Ollama LLM (phi3 model)...")
    llm = Ollama(
        model="phi3", 
        request_timeout=120.0,  # 2 minutes timeout for responses
        context_window=8000      # Maximum context size
    )
    
    # Create query engine with retrieval settings
    query_engine = index.as_query_engine(
        llm=llm, 
        similarity_top_k=3
    )
    
    # Interactive chatbot loop
    print("\n" + "="*60)
    print("🎉 Chatbot ready! Ask questions about your documents.")
    print("Type 'exit', 'quit', or 'sair' to close the chatbot.")
    print("="*60)
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() in ["sair", "exit", "quit"]:
            print("👋 Closing chatbot... Goodbye!")
            break
        if not question.strip():
            continue
        
        print("🔍 Searching documents...")
        response = query_engine.query(question)
        print(f"Bot: {response.response}")


if __name__ == "__main__":
    main()