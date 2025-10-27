from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.pinecone import PineconeVectorStore

from services.env_loader import EnvLoader
from services.pinecone_service import PineconeService

import arxiv
import re
from pathlib import Path


env = EnvLoader()

paper = next(arxiv.Client().results(arxiv.Search(id_list=["1603.09320"])))
paper.download_pdf(filename="hnsw.pdf")

loader = PDFReader()
documents = loader.load_data(file=Path("./hnsw.pdf"))
print(documents[0])


def clean_up_text(content: str) -> str:
    """
    Limpa e normaliza o texto extraído de PDFs.
    - Junta palavras quebradas por hífen no final da linha.
    - Remove caracteres e padrões indesejados (travessões, códigos unicode, quebras de linha literais).
    - Corrige espaços ao redor de hífens.
    - Substitui múltiplos espaços por um único espaço.
    Retorna o texto limpo e linearizado.
    """

    content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", content)
    unwanted_patterns = [
        "\\n",
        "  —",
        "——————————",
        "—————————",
        "—————",
        r"\\u[\dA-Fa-f]{4}",
        r"\uf075",
        r"\uf0b7",
    ]
    for pattern in unwanted_patterns:
        content = re.sub(pattern, "", content)
    content = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", content)
    content = re.sub(r"\s+", " ", content)
    return content


cleaned_docs = [
    Document(text=clean_up_text(d.text), metadata=d.metadata) for d in documents
]

print(cleaned_docs[0].get_content())

metadata_additions = {
    "authors": ["Yu. A. Malkov", "D. A. Yashunin"],
    "title": "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs",
}

[cd.metadata.update(metadata_additions) for cd in cleaned_docs]

print(cleaned_docs[0].metadata)


# Pinecone setup - 2025-10-27 11:47:11,988 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 429 Too Many Requests" 2025-10-27 11:47:11,989 - INFO - Retrying request to /embeddings in 0.409558 seconds
pc = PineconeService(api_key=env.PINECONE_API_KEY)
index_name = "llama-integration-example"
pc.ensure_index(index_name=index_name, dimension=768)

# Connect LlamaIndex to Pinecone
pinecone_index = pc.pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(cleaned_docs, storage_context=storage_context)

print("Documentos enviados ao Pinecone com sucesso!")
