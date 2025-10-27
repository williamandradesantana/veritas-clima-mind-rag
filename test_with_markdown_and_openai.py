from decouple import config
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import time
import os

OPENAI_API_KEY = config("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
PINECONE_API_KEY = config("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

markdown_document = """
## Introduction

The Earth is the third planet from the Sun and the only astronomical object known to harbor life. About 29.2% of Earth's surface is land with remaining 70.8% covered by water.
"""

headers_to_split_on = [("##", "Header 2")]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_headers_splits = markdown_splitter.split_text(markdown_document)

print(md_headers_splits)
print("\n")

pc = Pinecone(api_key=PINECONE_API_KEY)

cloud = config("PINECONE_CLOUD") or "aws"
region = config("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-getting-started"

model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(model=model_name, pinecone_api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, dimension=embeddings.dimension, metric="cosine", spec=spec
    )

# See that it is empty
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

namespace = "wondervector5000"
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
