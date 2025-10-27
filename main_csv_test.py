import pandas as pd
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from services.env_loader import EnvLoader

env = EnvLoader()

df = pd.read_csv("./data/spreadsheets/test-health-climate.csv")

documents = []
for _, row in df.iterrows():
    text = f"Data: {row['date']}. Temperatura média: {row['average_temperature']}°C. Umidade: {row['humidity']}%. Índice de ansiedade: {row['anxiety_index']}."
    documents.append(text)

pc = Pinecone(api_key=env.PINECONE_API_KEY)
spec = ServerlessSpec(cloud=env.PINECONE_CLOUD, region=env.PINECONE_REGION)

index_name = "health-climate-index"

model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(model=model_name, pinecone_api_key=env.PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec,
    )

namespace = "health-climate"

vectorstore = PineconeVectorStore.from_texts(
    texts=documents,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)

print("Dados inseridos no Pinecone!")
print(pc.Index(index_name).describe_index_stats())
