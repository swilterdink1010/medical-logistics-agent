import os

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

DB_DIR = "db"
DOCS_FILE = "data/medical_docs.txt"


def create_vector_db(docs_file=DOCS_FILE):
    if not os.path.exists(docs_file):
        raise FileNotFoundError(f"Missing docs file: {docs_file}")

    with open(docs_file, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("medical docs file is empty")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = Chroma.from_texts(docs, embeddings, persist_directory=DB_DIR)
    db.persist()
    return db

def load_vector_db():
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError("Vector DB not found. Run create_vector_db() first.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def rag_search(query, k=4):
    db = load_vector_db()
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])