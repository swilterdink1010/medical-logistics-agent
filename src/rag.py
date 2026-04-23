from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


def load_and_split()->list:
    loader = TextLoader("data/medical_docs.txt")
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
        persist_directory="./data/vectorstore"
    )
    
    
if __name__ == "__main__":
    ingest_chroma_db()

# def create_vector_db(docs_file=DOCS_FILE):
#     if not os.path.exists(docs_file):
#         raise FileNotFoundError(f"Missing docs file: {docs_file}")

#     with open(docs_file, "r", encoding="utf-8") as f:
#         text = f.read()

#     if not text.strip():
#         raise ValueError("medical docs file is empty")

#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.split_text(text)

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     db = Chroma.from_texts(docs, embeddings, persist_directory=DB_DIR)
#     db.persist()
#     return db

# def ensure_vectors():
    


# def load_vector_db():
#     if not os.path.exists(INDEX_NAME):
#         raise FileNotFoundError("Vector storage not found. Run create_vector_db() first.")

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return Chroma(persist_directory=INDEX_NAME, embedding_function=embeddings)


# def rag_search(query, k=4):
#     db = load_vector_db()
#     docs = db.similarity_search(query, k=k)
#     return "\n\n".join([d.page_content for d in docs])