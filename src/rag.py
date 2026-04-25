from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def load_and_split()->list:
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "data", "medical_docs.txt"))
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n"],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest_chroma_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    chunks = load_and_split()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="src/data/vectorstore"
    )
    
    
def load_vectorstore():
    ifndef_ingest()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="src/data/vectorstore",
    )
    return vectorstore


prompt_template = ChatPromptTemplate.from_template(
"""
You play the role of an AI assistant designed with the express purpose
of helping with medical logistics operations.

Answer the question using ONLY the context provided below. 
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:
"""
)
    
    
def create_rag_chain(llm):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain
    
    
def ifndef_ingest():
    if not os.path.exists("src/data/vectorstore/"):
        ingest_chroma_db()
    
    
if __name__ == "__main__":
    ingest_chroma_db()