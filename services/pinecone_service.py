from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time


class PineconeService:
    def __init__(self, api_key, cloud="aws", region="us-east-1"):
        self.pc = Pinecone(api_key=api_key)
        self.spec = ServerlessSpec(cloud=cloud, region=region)

    def ensure_index(self, index_name, dimension, metric="cosine"):
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name, dimension=dimension, metric=metric, spec=self.spec
            )
            print(f"🪴 Índice '{index_name}' criado.")
            time.sleep(2)

    def insert_texts(self, texts, index_name, embedding, namespace):
        PineconeVectorStore.from_texts(
            texts=texts,
            index_name=index_name,
            embedding=embedding,
            namespace=namespace,
        )
        print(f"✅ {len(texts)} textos inseridos em '{index_name}'.")

    def connect_to_index(self, index_name, embedding, namespace):
        return PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding,
            namespace=namespace,
        )

    def describe_index(self, index_name):
        print(self.pc.Index(index_name).describe_index_stats())
