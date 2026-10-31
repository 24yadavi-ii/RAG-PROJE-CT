import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction")

# -----------------------------
# Cache Resource
# -----------------------------
@st.cache_resource
def load_vectorstore():
    loader = TextLoader(
        "C++_Introduction.txt",
        encoding="utf-8"
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(final_documents, embeddings)
    return db

# -----------------------------
# Load Vector Database
# -----------------------------
try:
    db = load_vectorstore()
except Exception as error:
    st.error(f"Error loading documents: {error}")
    st.stop()

# -----------------------------
# Load LLM using Ollama
# -----------------------------
try:
    llm = Ollama(model="gemma2:2b")
except Exception as error:
    st.error(f"Error initializing Ollama: {error}")
    st.stop()

# -----------------------------
# Chat Interface
# -----------------------------
user_question = st.text_input(
    "Ask a question about C++:"
)

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            docs = db.similarity_search(user_question, k=3)
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are a helpful C++ assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say:
"I could not find this information in the provided document."

Context:
{context}

Question:
{user_question}

Answer:
"""

            try:
                response = llm.invoke(prompt)
                st.subheader("Answer")
                st.write(response)

                st.subheader("Sources")
                for doc in docs:
                    st.write(doc.metadata.get("source", "Unknown source"))
                    st.caption(doc.page_content)

            except Exception as error:
                st.error(
                    f"Could not generate an answer: {error}. "
                    "Check that Ollama is running and gemma2:2b is installed."
                )


