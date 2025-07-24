from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load local LLaMA 3 from Ollama
def load_llm_ollama():
    
    return Ollama(model="mistral")

# Step 2: Custom prompt for better context usage
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know. Do not make up answers.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS DB built from JSON
DB_FAISS_PATH = "vectorstore/db_faiss_json"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm_ollama(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Run chat
user_query = input("💬 Ask your IT support question: ")
response = qa_chain.invoke({'query': user_query})

# Step 6: Show results
print("\n🧠 Answer:\n", response["result"])
print("\n📚 Source Documents:\n")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"[{i}] {doc.page_content[:300]}...\n---\n")
