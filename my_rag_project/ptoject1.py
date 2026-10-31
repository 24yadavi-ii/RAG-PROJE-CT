
import re
from pathlib import Path

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ================= CONFIGURATION =================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR / "documents"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 120
CHUNK_OVERLAP = 30
TOP_K = 5
MIN_SCORE = 0.30


st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)


# ================= LOAD DOCUMENTS =================

def load_documents(folder):
    records = []
    errors = []

    folder.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(folder.glob("*.txt")) +
        list(folder.glob("*.pdf"))
    )

    for file_path in files:
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(
                    encoding="utf-8-sig",
                    errors="replace"
                )

                if text.strip():
                    records.append({
                        "text": text,
                        "source": file_path.name,
                        "page": "N/A"
                    })
                else:
                    errors.append(
                        f"{file_path.name}: file is empty."
                    )

            elif file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))
                extracted_pages = 0

                for page_number, page in enumerate(
                    reader.pages, start=1
                ):
                    text = page.extract_text() or ""

                    if text.strip():
                        records.append({
                            "text": text,
                            "source": file_path.name,
                            "page": page_number
                        })
                        extracted_pages += 1

                if extracted_pages == 0:
                    errors.append(
                        f"{file_path.name}: no extractable text. "
                        "A scanned PDF may require OCR."
                    )

        except Exception as exc:
            errors.append(
                f"{file_path.name}: {type(exc).__name__}: {exc}"
            )

    return records, errors


# ================= TEXT CHUNKING =================

def make_chunks(records):
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for record in records:
        words = record["text"].split()

        for start in range(0, len(words), step):
            part = " ".join(
                words[start:start + CHUNK_SIZE]
            )

            if len(part.strip()) < 15:
                continue

            chunks.append({
                "text": part,
                "source": record["source"],
                "page": record["page"]
            })

    return chunks


# ================= EMBEDDING MODEL =================

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


# ================= DOCUMENT REFRESH =================

@st.cache_data(show_spinner=False)
def get_file_signature():
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

    return tuple(
        (
            path.name,
            path.stat().st_size,
            path.stat().st_mtime_ns
        )
        for path in sorted(DOCUMENT_DIR.iterdir())
        if path.is_file()
        and path.suffix.lower() in {".txt", ".pdf"}
    )


def build_knowledge_base(model, signature):
    records, errors = load_documents(DOCUMENT_DIR)
    chunks = make_chunks(records)

    if chunks:
        embeddings = model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )
    else:
        embeddings = np.empty(
            (0, 0),
            dtype=np.float32
        )

    return records, chunks, embeddings, errors


# ================= RETRIEVAL =================

def retrieve(question, chunks, embeddings, model):
    if not chunks:
        return []

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in indices:
        item = dict(chunks[int(index)])
        item["score"] = float(scores[index])
        results.append(item)

    return results


# ================= ANSWER HELPERS =================

def extract_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text
        )
        if len(sentence.strip()) > 15
    ]


def find_numeric_conflict(relevant):
    """
    Basic heuristic:
    Detect different percentages or numbers in relevant
    passages from different documents.

    This is not a complete contradiction detector.
    """

    claims = {}

    for item in relevant:
        numbers = re.findall(
            r"\b\d+(?:\.\d+)?\s*%?",
            item["text"]
        )

        if numbers:
            claims.setdefault(
                item["source"], set()
            ).update(
                number.strip() for number in numbers
            )

    sources = list(claims.keys())

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            source_a = sources[i]
            source_b = sources[j]

            # Find whether both documents discuss percentages.
            percentages_a = {
                n for n in claims[source_a] if "%" in n
            }
            percentages_b = {
                n for n in claims[source_b] if "%" in n
            }

            if (
                percentages_a
                and percentages_b
                and percentages_a != percentages_b
            ):
                return source_a, percentages_a, source_b, percentages_b

    return None


# ================= ANSWER GENERATION =================

def make_answer(question, results, model):
    relevant = [
        item for item in results
        if item["score"] >= MIN_SCORE
    ]

    if not relevant:
        return (
            "I could not find sufficiently relevant information "
            "in the available documents. Please consult the "
            "appropriate college department."
        ), [], False

    # Check for a possible conflict.
    conflict_details = find_numeric_conflict(relevant)

    if conflict_details:
        source_a, values_a, source_b, values_b = conflict_details

        # Normal answer, not a warning box.
        answer = (
            "The retrieved documents provide different numeric "
            "requirements. "
            f"{source_a} states {', '.join(sorted(values_a))}, "
            f"whereas {source_b} states "
            f"{', '.join(sorted(values_b))}. "
            "Please verify the current official policy."
        )

        return answer, relevant, False

    # Collect sentences from relevant passages.
    candidates = []

    for item in relevant:
        for sentence in extract_sentences(item["text"]):
            candidates.append({
                "sentence": sentence,
                "source": item["source"],
                "page": item["page"],
                "score": item["score"]
            })

    # Remove duplicate sentences.
    unique = {}

    for item in candidates:
        key = item["sentence"].lower()

        if (
            key not in unique
            or item["score"] > unique[key]["score"]
        ):
            unique[key] = item

    candidates = list(unique.values())

    if not candidates:
        return (
            "The retrieved documents do not provide a clear answer."
        ), relevant, False

    # Rank sentences by semantic similarity to the question.
    sentence_embeddings = model.encode(
        [item["sentence"] for item in candidates],
        normalize_embeddings=True
    )

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    sentence_scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    for i, item in enumerate(candidates):
        item["sentence_score"] = float(sentence_scores[i])

    candidates.sort(
        key=lambda item: item["sentence_score"],
        reverse=True
    )

    # Return up to four evidence-based sentences.
    selected = candidates[:4]

    answer = "\n\n".join(
        item["sentence"] for item in selected
    )

    return answer, relevant, False


# ================= USER INTERFACE =================

st.title("🎓 College Knowledge Assistant")

st.write(
    "Ask questions about college documents. "
    "Answers are based on retrieved document passages."
)

st.caption(
    "RAG Pipeline: Document Loading → Chunking → "
    "Embeddings → Semantic Retrieval → Evidence-based Answer"
)


# Sidebar
with st.sidebar:
    st.header("Knowledge Base")

    st.write("Document folder:")
    st.code(str(DOCUMENT_DIR), language=None)

    st.write(f"Top-K retrieval: {TOP_K}")
    st.write(f"Minimum similarity: {MIN_SCORE}")

    if st.button("Refresh documents"):
        get_file_signature.clear()
        st.rerun()

    st.info(
        "Add PDF or TXT files to the documents folder, "
        "then click Refresh documents."
    )


# Load model
try:
    with st.spinner("Loading embedding model..."):
        model = load_model()

except Exception as exc:
    st.error(
        "Could not load the embedding model. "
        "Check your internet connection and installation."
    )
    st.exception(exc)
    st.stop()


# Build knowledge base
signature = get_file_signature()

try:
    with st.spinner("Processing college documents..."):
        records, chunks, embeddings, errors = (
            build_knowledge_base(model, signature)
        )

except Exception as exc:
    st.error("Could not build the knowledge base.")
    st.exception(exc)
    st.stop()


# Document loading errors
if errors:
    with st.expander("Document loading details"):
        for error in errors:
            st.write(error)


# Statistics
document_count = len({
    record["source"] for record in records
})

col1, col2, col3 = st.columns(3)

col1.metric(
    "Documents with extracted text",
    document_count
)

col2.metric(
    "Searchable chunks",
    len(chunks)
)

col3.metric(
    "Minimum required files",
    5
)


if document_count < 5:
    st.info(
        "Add at least five readable PDF/TXT documents "
        "to meet the project requirement."
    )


if not chunks:
    st.error(
        "No searchable text found. Add readable TXT or PDF "
        "documents to the documents folder."
    )
    st.stop()


st.divider()


# Question input
question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What attendance is required for exams?"
    )
)


if st.button("Get Answer", type="primary"):

    if not question.strip():
        st.info("Please enter a question first.")

    else:
        with st.spinner("Searching college documents..."):

            results = retrieve(
                question,
                chunks,
                embeddings,
                model
            )

            answer, evidence, conflict = make_answer(
                question,
                results,
                model
            )

        st.subheader("Answer")

        # Always display the answer as normal text.
        # No yellow conflict-warning box is used.
        st.write(answer)

        st.subheader("Supporting Sources")

        if evidence:
            for number, item in enumerate(
                evidence, start=1
            ):
                title = (
                    f"{number}. {item['source']} | "
                    f"Page: {item['page']} | "
                    f"Similarity: {item['score']:.3f}"
                )

                with st.expander(title):
                    st.write(item["text"])

                    st.caption(
                        "Similarity is a retrieval score, "
                        "not a guarantee of factual correctness."
                    )
        else:
            st.write(
                "No sufficiently relevant sources were found."
            )


st.divider()

st.caption(
    "Demo application. Verify important academic policies "
    "against current official college documents."
)

