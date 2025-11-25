import os

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.pinecone import PineconeVectorStore

from services.env_loader import EnvLoader
from services.pinecone_service import PineconeService
from chatbot import gather_documents
from services.behavior_analysis_services import analyze_behavior_text

import datetime


def main():
    env = EnvLoader()

    # Initialize Pinecone service
    pc = PineconeService(api_key=env.PINECONE_API_KEY)

    index_name = "scientific-basis-example"
    namespace = "default"

    pc.ensure_index(index_name=index_name, dimension=768)

    # Connect to existing Pinecone index
    print(f"🌦️ Connecting to Pinecone index '{index_name}'...")
    pinecone_index = pc.pc.Index(index_name)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace=namespace
    )

    # Embedding model (using local Ollama)
    embedding_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if there are already vectors in the index
    stats = pinecone_index.describe_index_stats()
    total_vectors = (
        stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    )

    if total_vectors == 0:
        print(f"📂 Index '{index_name}' is empty. Indexing 'news' PDFs...")

        documents = gather_documents("data", subfolder="scientific_basis")

        if documents:
            print(f"🔧 Indexing {len(documents)} weather documents...")
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=embedding_model
            )
            print(f"✅ {len(documents)} documents indexed successfully!")
        else:
            print("⚠️ No documents found to index.")
            return
    else:
        print(f"📊 Loading existing index with {total_vectors} vectors...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embedding_model
        )
        print("✅ Index loaded successfully!")

    print("🤖 Initializing Ollama model (phi3)...")
    llm = Ollama(
        model="phi3",
        request_timeout=100020.0,
        context_window=8000,
    )

    # Create query engine
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

    print("\n" + "=" * 60)
    print("Chat is ready!")
    print("Ask about climate change and psychology.")
    print("Type 'exit' or 'quit' to finish.")
    print("=" * 60)

    while True:
        print("\n🎤 Enter your question")
        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            print("👋 Shutting down the assistant. See you!")
            break

        if not question.strip():
            continue

        response = query_engine.query(question)
        answer = response.response

        print(f"Assistant: {answer}")

        print("🧠 Evaluating behavioral markers of the response...")
        markers = analyze_behavior_text(answer)
        print("📊 Markers:", markers)

        os.makedirs("logs", exist_ok=True)
        log_file = "logs/behavior_log_scientific_basis.txt"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"📅 Date: {datetime.datetime.now()}\n")
            f.write(f"🎤 Question: {question}\n")
            f.write(f"🤖 AI Response: {answer}\n")
            f.write(f"🧠 Behavioral markers: {markers}\n")

        print(f"💾 Log saved to {log_file}")


if __name__ == "__main__":
    main()
