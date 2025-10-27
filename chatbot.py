from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

import time

from test_with_markdown_and_gemini import *

docsearch = PineconeVectorStore(
    index_name=index_name, embedding=embeddings, namespace=namespace
)

# Criar prompt personalizado para RAG
retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and to the point.

Context: {context}""",
        ),
        ("human", "{input}"),
    ]
)

# Configurar retriever
retriever = docsearch.as_retriever(
    search_kwargs={"k": 3}  # Retorna os 3 documentos mais relevantes
)

# Configurar o modelo Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # ou "gemini-1.5-pro" para melhor qualidade
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True,  # Importante para Gemini
)

# Criar as chains
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Query 1
query1 = "What are the first 3 steps for getting started with the WonderVector5000?"
print("=" * 80)
print("QUERY 1:", query1)
print("=" * 80)

# Resposta SEM conhecimento (sem RAG)
print("\n[WITHOUT RAG - Direct Gemini Response]")
answer1_without_knowledge = llm.invoke(query1)
print(answer1_without_knowledge.content)

time.sleep(1)

# Resposta COM conhecimento (com RAG)
print("\n[WITH RAG - Context-Aware Response]")
answer1_with_knowledge = retrieval_chain.invoke({"input": query1})
print(answer1_with_knowledge["answer"])
print("\n")

time.sleep(2)

# Query 2
query2 = "The Neural Fandango Synchronizer is giving me a headache. What do I do?"
print("=" * 80)
print("QUERY 2:", query2)
print("=" * 80)

# Resposta SEM conhecimento
print("\n[WITHOUT RAG - Direct Gemini Response]")
answer2_without_knowledge = llm.invoke(query2)
print(answer2_without_knowledge.content)

time.sleep(1)

# Resposta COM conhecimento
print("\n[WITH RAG - Context-Aware Response]")
answer2_with_knowledge = retrieval_chain.invoke({"input": query2})
print(answer2_with_knowledge["answer"])

# Mostrar os documentos recuperados
print("\n" + "=" * 80)
print("RETRIEVED DOCUMENTS")
print("=" * 80)
for i, doc in enumerate(answer2_with_knowledge["context"], 1):
    print(f"\n📄 Document {i}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
