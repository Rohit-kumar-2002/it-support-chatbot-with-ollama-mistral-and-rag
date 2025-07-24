import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="💼 IT Support Assistant",
    layout="wide",
    page_icon="💬"
)

# ------------------- INLINE CUSTOM THEME STYLING -------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #e7dff5;
    color: #b97575;
    font-family: 'Segoe UI', sans-serif;
}

.css-1rs6os.edgvbvh3 {
    background-color: #d8e4e4 !important;
}

.chat-container {
    padding: 1.5rem;
    background-color: #d8e4e4;
    border-radius: 16px;
    max-width: 85%;
    margin: auto;
}
.message {
    padding: 1rem;
    margin-bottom: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.message.user {
    background-color: #f7dbdb;
    border-left: 6px solid #c59999;
    text-align: right;
}
.message.bot {
    background-color: #f1ecfa;
    border-left: 6px solid #c59999;
    text-align: left;
}
.source {
    font-size: 0.85rem;
    color: #444;
    background: #f5f5f5;
    border-left: 3px solid #b97575;
    padding: 0.5rem 1rem;
    margin-top: 0.75rem;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- SETTINGS SIDEBAR -------------------
st.sidebar.title("⚙️ Chat Settings")
st.sidebar.markdown("Customize chatbot behavior")

k_value = st.sidebar.slider("🔍 Context Chunks (k)", min_value=1, max_value=10, value=3)

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ------------------- LOAD VECTORSTORE -------------------
DB_FAISS_PATH = "vectorstore/db_faiss_json"

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

retriever = load_vectorstore().as_retriever(search_kwargs={"k": k_value})

# ------------------- PROMPT TEMPLATE -------------------
CUSTOM_PROMPT_TEMPLATE = """
You are an expert IT assistant in a tech company.
Use the context below to respond helpfully and clearly.
If unsure, say "I don't know"—do not guess.

Context:
{context}

Question:
{question}

Answer:
"""

def get_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# ------------------- LLM & QA CHAIN -------------------
llm = Ollama(model="llama3")  # or mistral if preferred

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": get_prompt()}
)

# ------------------- SESSION STATE -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- HEADER & INSTRUCTIONS -------------------
st.title("💬 IT Support Assistant")
st.caption("Ask anything about your IT setup, system issues, integrations, or infrastructure.")

with st.expander("💡 How to use", expanded=False):
    st.markdown("""
- Example prompts:
  - "Why is my dashboard crashing after update?"
  - "How to protect medical data on Windows?"
  - "PostgreSQL backup strategy in AWS?"
- This chatbot works **fully offline** (Ollama + LangChain).
""")

# ------------------- CHAT INPUT -------------------
user_input = st.chat_input("Type your IT issue or question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("🤔 Thinking..."):
        try:
            response = qa_chain.invoke({"query": user_input})
            answer = response["result"]
            sources = response["source_documents"]

            st.session_state.chat_history.append({
                "role": "bot",
                "content": answer,
                "sources": sources
            })
        except Exception as e:
            st.error("⚠️ Something went wrong: " + str(e))

# ------------------- DISPLAY CHAT -------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    css_class = "user" if role == "user" else "bot"
    label = "You" if role == "user" else "Bot"

    st.markdown(f'<div class="message {css_class}"><b>{label}:</b><br>{content}</div>', unsafe_allow_html=True)

    if role == "bot" and "sources" in msg:
        with st.expander("📚 Show Sources"):
            for i, doc in enumerate(msg["sources"], 1):
                st.markdown(
                    f"""<div class="source"><b>Source [{i}]</b>:<br>{doc.page_content[:300]}...</div>""",
                    unsafe_allow_html=True
                )

st.markdown('</div>', unsafe_allow_html=True)
