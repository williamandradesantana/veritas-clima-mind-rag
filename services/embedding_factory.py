from langchain_pinecone import PineconeEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingFactory:
    @staticmethod
    def create(provider: str, api_key: str):
        provider = provider.lower()
        if provider == "openai":
            return PineconeEmbeddings(
                model="multilingual-e5-large", pinecone_api_key=api_key
            )
        elif provider == "google":
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=api_key
            )
        else:
            raise ValueError("Embedding não suportado: use 'openai' ou 'google'.")
