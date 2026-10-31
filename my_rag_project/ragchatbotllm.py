
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction.")

# -----------------------------
# Find Text File
# -----------------------------
@st.cache_resource
def load_vectorstore():
    current_folder = Path(__file__).resolve().parent

    possible_paths = [
        current_folder / "C++_Introduction.txt",
        current_folder.parent / "C++_Introduction.txt",
        current_folder / "documents" / "C++_Introduction.txt",
        current_folder.parent / "documents" / "C++_Introduction.txt",
    ]

    file_path = next(
        (path for path in possible_paths if path.is_file()),
        None
    )

    if file_path is None:
        raise FileNotFoundError(
            "C++_Introduction.txt was not found. "
            "Place it in the project folder or documents folder."
        )

    # Load document
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )
    documents = loader.load()

    if not documents or not documents[0].page_content.strip():
        raise ValueError("The text file is empty.")

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    final_documents = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create searchable vector database
    db = FAISS.from_documents(final_documents, embeddings)

    return db


# -----------------------------
# Load Vector Database
# -----------------------------
try:
    with st.spinner("Loading C++ document and building knowledge base..."):
        db = load_vectorstore()

    st.success("C++ knowledge base loaded successfully!")

except Exception as error:
    st.error(f"Error loading documents: {error}")
    st.stop()


# -----------------------------
# Load Ollama LLM
# -----------------------------
try:
    llm = OllamaLLM(model="gemma2:2b")

except Exception as error:
    st.error(f"Error initializing Ollama: {error}")
    st.stop()


# -----------------------------
# Chat Interface
# -----------------------------
user_question = st.text_input(
    "Ask a question about C++:",
    placeholder="e.g. What is C++?"
)

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching documents and generating answer..."):

                # Search relevant chunks
                docs = db.similarity_search(user_question, k=3)

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                # RAG prompt
                prompt = f"""
You are a helpful C++ assistant.

Answer the question using ONLY the context below.
Do not use outside knowledge or invent facts.

If the answer is not present in the context, say:
"I could not find this information in the provided document."

Context:
{context}

Question:
{user_question}

Answer:
"""

                response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response)

            st.subheader("Sources")
            for doc in docs:
                st.write(
                    doc.metadata.get("source", "Unknown source")
                )
                st.caption(doc.page_content)

        except Exception as error:
            st.error(f"Could not generate an answer: {error}")
            st.info(
                "Check that Ollama is running and the gemma2:2b "
                "model is installed."
            )
