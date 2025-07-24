from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load JSON
DATA_PATH = "D:/ollama/project/data/rag_ready_qa_dataset.json"

def load_json_file(path):
    loader = JSONLoader(
        file_path=path,
        jq_schema='.[] | {page_content: (.question + " " + .answer), metadata: {language: .language}}',
        text_content=False
    )
    return loader.load()

documents = load_json_file(DATA_PATH)

# Step 2: Chunk
def create_chunks(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(data)

text_chunks = create_chunks(documents)

# Step 3: Embed
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss_json"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
