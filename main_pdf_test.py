from services.env_loader import EnvLoader
from services.embedding_factory import EmbeddingFactory
from services.pinecone_service import PineconeService
from loaders.pdf_loader import PDFLoader

env = EnvLoader()

embeddings = EmbeddingFactory.create("openai", env.PINECONE_API_KEY)

pinecone_service = PineconeService(
    env.PINECONE_API_KEY, env.PINECONE_CLOUD, env.PINECONE_REGION
)

index_name = "health-climate-pdf-index"
namespace = "health-climate-pdf-climate-distress"

pinecone_service.ensure_index(index_name, embeddings.dimension)

loader = PDFLoader(
    "./data/pdfs/Climate_Distress_A_Review_of_Current_Psychological_Research_and_Practice.pdf"
)
texts = loader.extract_text_chunks()

pinecone_service.insert_texts(texts, index_name, embeddings, namespace)
pinecone_service.describe_index(index_name)

# Consulta
vectorstore = pinecone_service.connect_to_index(index_name, embeddings, namespace)
query = "O documento fala sobre o impacto do calor na saúde mental?"
results = vectorstore.similarity_search(query, k=3)

for i, r in enumerate(results, start=1):
    print(f"\n Resultado {i}:\n{r.page_content[:400]}...")
