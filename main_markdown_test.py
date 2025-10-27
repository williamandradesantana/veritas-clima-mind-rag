from services.env_loader import EnvLoader
from services.embedding_factory import EmbeddingFactory
from services.pinecone_service import PineconeService
from loaders.markdown_loader import MarkdownLoader

env = EnvLoader()

embeddings = EmbeddingFactory.create("openai", env.OPENAI_API_KEY)

pinecone_service = PineconeService(
    api_key=env.PINECONE_API_KEY,
    cloud=env.PINECONE_CLOUD,
    region=env.PINECONE_REGION,
)

index_name = "markdown-demo-index"
namespace = "markdown-namespace2"

pinecone_service.ensure_index(index_name, dimension=768)  # Google embeddings

markdown_text = """
## Introdução
O planeta Terra é o único conhecido por abrigar vida.

## Como começar
1. Ligue o dispositivo.
2. Conecte-se à energia.

## Solução de problemas
Se encontrar erros, reinicie o sistema.
"""

loader = MarkdownLoader(markdown_text)
docs = loader.split()

pinecone_service.insert_texts(
    texts=[doc.page_content for doc in docs],
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)

pinecone_service.describe_index(index_name)

# Consulta o índice
vectorstore = pinecone_service.connect_to_index(index_name, embeddings, namespace)
query = "O texto fala sobre como resolver problemas?"
results = vectorstore.similarity_search(query, k=2)

for i, r in enumerate(results, start=1):
    print(f"\n🔹 Resultado {i}:\n{r.page_content[:300]}...")
