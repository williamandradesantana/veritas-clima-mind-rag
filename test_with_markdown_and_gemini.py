from decouple import config
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import time
import os

# Configuração das API Keys
GOOGLE_API_KEY = config("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PINECONE_API_KEY = config("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Documento markdown de exemplo
markdown_document = """
# Introduction

The Earth is the third planet from the Sun and the only astronomical object known to harbor life. About 29.2% of Earth's surface is land with remaining 70.8% covered by water.

## Getting Started with the WonderVector5000

To get started with the WonderVector5000, follow these steps:

1. Unbox the WonderVector5000 and ensure all components are present.
2. Connect the device to a power source using the provided adapter.
3. Power on the device and follow the on-screen setup instructions to configure your preferences and connect to Wi-Fi.

### Troubleshooting Common Issues

If you encounter issues with the Neural Fandango Synchronizer, try the following troubleshooting steps:

1. Restart the device to reset any temporary glitches.
2. Ensure that your firmware is up to date by checking for updates in the settings menu.
3. If problems persist, contact WonderVector support for further assistance."""

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_headers_splits = markdown_splitter.split_text(markdown_document)

# Configuração do Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = config("PINECONE_CLOUD", default="aws")
region = config("PINECONE_REGION", default="us-east-1")
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-gemini-demo"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)

# Criar índice se não existir
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=spec,
    )
    time.sleep(1)

# Ver estatísticas antes do upsert
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

namespace = "wondervector5000"

# Inserir documentos no Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=md_headers_splits,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)

print("Index after upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

time.sleep(2)
print("Setup completed successfully!")
