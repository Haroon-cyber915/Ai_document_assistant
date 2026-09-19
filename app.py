
"""
AI Document Assistant
----------------------
A simple Streamlit app that lets you upload documents (PDF, DOCX, TXT, MD)
or load them from Google Drive, then ask questions answered strictly from
their content using hybrid (semantic + keyword) search and Groq's LLM.
"""
 
import io
import re
 
import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
 
 
# =========================================================================
# Page setup
# =========================================================================
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")
 
 
# =========================================================================
# Cached resources (created once per session, reused afterwards)
# =========================================================================
@st.cache_resource
def load_embedding_model():
    """Load the sentence-transformer model once and reuse it."""
    return SentenceTransformer("all-MiniLM-L6-v2")
 
 
@st.cache_resource
def get_groq_client():
    """Create the Groq client using the key from Streamlit secrets."""
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)
 
 
# =========================================================================
# Document extraction (one function per file type, plus a dispatcher)
# =========================================================================
def extract_pdf(file_bytes, filename):
    """Extract text from a PDF, page by page. Keeps page numbers."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"filename": filename, "page": i, "text": text})
    return pages
 
 
def extract_docx(file_bytes, filename):
    """Extract text from a DOCX file. Word files have no fixed pages, so page=None."""
    document = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [{"filename": filename, "page": None, "text": text}] if text.strip() else []
 
 
def extract_txt(file_bytes, filename):
    """Extract text from a plain TXT file."""
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"filename": filename, "page": None, "text": text}] if text.strip() else []
 
 
def extract_md(file_bytes, filename):
    """Extract text from a Markdown file (treated as plain text)."""
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"filename": filename, "page": None, "text": text}] if text.strip() else []
 
 
def extract_document(filename, file_bytes):
    """Pick the right extractor based on the file extension."""
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_pdf(file_bytes, filename)
    if ext == "docx":
        return extract_docx(file_bytes, filename)
    if ext == "txt":
        return extract_txt(file_bytes, filename)
    if ext == "md":
        return extract_md(file_bytes, filename)
    return []
 
 
# =========================================================================
# Chunking
# =========================================================================
def chunk_text(pages, chunk_size=180, overlap=40):
    """
    Split extracted pages into overlapping word chunks.
    Every chunk keeps the filename, page number, and its index within the page.
    """
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        start = 0
        chunk_index = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append({
                "filename": page["filename"],
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": " ".join(chunk_words),
            })
            chunk_index += 1
            start += chunk_size - overlap
    return chunks
 
 
# =========================================================================
# Embeddings + FAISS index
# =========================================================================
def embed_chunks(chunks, model):
    """Create normalized embeddings for a list of chunks (for cosine similarity)."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")
 
 
def build_faiss_index(embeddings):
    """Build a FAISS index. Inner product on normalized vectors == cosine similarity."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index
 
 
# =========================================================================
# Search: semantic, keyword, and hybrid
# =========================================================================
def semantic_search(query, model, index, chunks, top_k=5):
    """Embed the query and retrieve the most semantically similar chunks."""
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"chunk": chunks[idx], "score": float(score)})
    return results
 
 
STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "of", "to", "in",
    "for", "with", "what", "how", "does", "do", "are", "was", "were", "this",
    "that", "it", "as", "by", "be", "can", "from",
}
 
 
def keyword_search(query, chunks, top_k=5):
    """Score chunks by how many important query words they contain."""
    keywords = [w.lower() for w in re.findall(r"\w+", query) if w.lower() not in STOPWORDS]
    if not keywords:
        return []
    results = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        matches = sum(text_lower.count(k) for k in keywords)
        if matches > 0:
            results.append({"chunk": chunk, "score": matches / len(keywords)})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
 
 
def _normalize_scores(results):
    """Scale a list of {chunk, score} results to the [0, 1] range."""
    if not results:
        return {}
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    normalized = {}
    for r in results:
        key = id(r["chunk"])
        normalized[key] = (r["score"] - min_s) / (max_s - min_s) if max_s > min_s else 1.0
    return normalized
 
 
def hybrid_search(query, model, index, chunks, top_k=5, alpha=0.7):
    """
    Combine semantic and keyword search into one ranked list.
    alpha controls the weight given to semantic search vs keyword search.
    """
    semantic_results = semantic_search(query, model, index, chunks, top_k=len(chunks))
    keyword_results = keyword_search(query, chunks, top_k=len(chunks))
 
    sem_norm = _normalize_scores(semantic_results)
    kw_norm = _normalize_scores(keyword_results)
 
    combined = {}
    for r in semantic_results:
        key = id(r["chunk"])
        combined[key] = {"chunk": r["chunk"], "score": alpha * sem_norm.get(key, 0)}
    for r in keyword_results:
        key = id(r["chunk"])
        if key in combined:
            combined[key]["score"] += (1 - alpha) * kw_norm.get(key, 0)
        else:
            combined[key] = {"chunk": r["chunk"], "score": (1 - alpha) * kw_norm.get(key, 0)}
 
    ranked = sorted(combined.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]
 
 
# =========================================================================
# Groq answer generation
# =========================================================================
def generate_answer(client, query, retrieved):
    """Ask Groq's LLM to answer using only the retrieved context."""
    if client is None:
        return "Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."
 
    context_parts = []
    for r in retrieved:
        c = r["chunk"]
        page_info = f", page {c['page']}" if c["page"] else ""
        context_parts.append(f"[Source: {c['filename']}{page_info}]\n{c['text']}")
    context = "\n\n".join(context_parts)
 
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided document context. If the answer is not contained in the "
        "context, clearly say the information is not available in the "
        "documents. Do not use outside knowledge."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
 
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
 
 
# =========================================================================
# Google Drive helpers
# =========================================================================
def parse_drive_link(url):
    """Return ('file', id), ('folder', id), or (None, None)."""
    folder_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if folder_match:
        return "folder", folder_match.group(1)
    file_match = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if file_match:
        return "file", file_match.group(1)
    return None, None
 
 
def download_drive_file(file_id):
    """Download a publicly shared Google Drive file. Returns (filename, bytes)."""
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True)
 
    # Large files show a virus-scan warning page with a confirm token in a cookie.
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)
 
    if response.status_code != 200:
        return None, None
 
    filename = None
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        filename = match.group(1)
 
    return filename, response.content
 
 
def list_drive_folder(folder_id, api_key):
    """List files in a public Drive folder using the Drive API (needs an API key)."""
    from googleapiclient.discovery import build
 
    service = build("drive", "v3", developerKey=api_key)
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get("files", [])
 
 
# =========================================================================
# Session state
# =========================================================================
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = set()  # (filename, size) already embedded
if "doc_info" not in st.session_state:
    st.session_state.doc_info = []  # per-document stats shown in the UI
 
model = load_embedding_model()
groq_client = get_groq_client()
 
st.title("📄 AI Document Assistant")
st.caption("Upload documents or load them from Google Drive, then ask questions grounded in their content.")
 
if groq_client is None:
    st.warning(
        "GROQ_API_KEY is not set. Search will still work, but no answer can be "
        "generated. Add the key under Manage app → Settings → Secrets."
    )
 
 
# =========================================================================
# Sidebar: add documents
# =========================================================================
with st.sidebar:
    st.header("1. Add documents")
 
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
 
    st.markdown("---")
    st.subheader("Google Drive")
    drive_link = st.text_input("Paste a Drive file or folder link")
    google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    load_drive_clicked = st.button("Load from Drive")
 
    st.markdown("---")
    process_clicked = st.button("Process documents", type="primary")
 
    st.markdown("---")
    st.subheader("2. Search settings")
    alpha = st.slider(
        "Semantic vs keyword weight", 0.0, 1.0, 0.7, 0.1,
        help="Higher = rely more on meaning (semantic search), lower = rely more on exact words.",
    )
    top_k = st.slider("Chunks to retrieve per question", 1, 10, 5)
 
new_files = []  # list of (filename, bytes) collected this run
 
if uploaded_files:
    for f in uploaded_files:
        # getvalue() (not read()) — read() empties the buffer on Streamlit reruns
        new_files.append((f.name, f.getvalue()))
 
if load_drive_clicked and drive_link:
    kind, drive_id = parse_drive_link(drive_link)
    if kind == "file":
        filename, content = download_drive_file(drive_id)
        if content and filename and filename.lower().split(".")[-1] in ("pdf", "docx", "txt", "md"):
            new_files.append((filename, content))
            st.sidebar.success(f"Loaded '{filename}' from Drive.")
        else:
            st.sidebar.error("Could not download a supported file. Make sure the link is public.")
    elif kind == "folder":
        if not google_api_key:
            st.sidebar.error("Loading a folder needs GOOGLE_API_KEY in Streamlit secrets.")
        else:
            try:
                files_meta = list_drive_folder(drive_id, google_api_key)
                loaded_count = 0
                for meta in files_meta:
                    ext = meta["name"].lower().split(".")[-1]
                    if ext in ("pdf", "docx", "txt", "md"):
                        _, content = download_drive_file(meta["id"])
                        if content:
                            new_files.append((meta["name"], content))
                            loaded_count += 1
                st.sidebar.success(f"Loaded {loaded_count} file(s) from the Drive folder.")
            except Exception as e:
                st.sidebar.error(f"Could not read the Drive folder: {e}")
    else:
        st.sidebar.error("That link doesn't look like a Drive file or folder link.")
 
 
# =========================================================================
# Processing: extract -> chunk -> embed -> index
# (only new, not-yet-seen files are processed, so we never re-embed on
# every question or every rerun)
# =========================================================================
if process_clicked:
    if not new_files:
        st.sidebar.warning("Add at least one file first.")
    else:
        fresh_chunks = []
        with st.spinner("Extracting and chunking documents..."):
            for filename, content in new_files:
                source_key = (filename, len(content))
                if source_key in st.session_state.processed_sources:
                    continue  # already embedded earlier, skip
                if not content:
                    st.sidebar.error(f"'{filename}' came through empty. Re-upload it and try again.")
                    continue
                pages = extract_document(filename, content)
                if not pages:
                    st.sidebar.warning(
                        f"No text could be extracted from '{filename}'. "
                        "If it's a scanned PDF, it has no selectable text layer."
                    )
                    continue
                doc_chunks = chunk_text(pages)
                fresh_chunks.extend(doc_chunks)
                st.session_state.processed_sources.add(source_key)
                st.session_state.doc_info.append({
                    "filename": filename,
                    "pages": len(pages),
                    "chunks": len(doc_chunks),
                })
 
        if fresh_chunks:
            with st.spinner(f"Creating embeddings for {len(fresh_chunks)} chunks..."):
                new_embeddings = embed_chunks(fresh_chunks, model)
                st.session_state.all_chunks.extend(fresh_chunks)
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = new_embeddings
                else:
                    st.session_state.embeddings = np.vstack([st.session_state.embeddings, new_embeddings])
                st.session_state.faiss_index = build_faiss_index(st.session_state.embeddings)
            st.sidebar.success(f"Added {len(fresh_chunks)} new chunks.")
        else:
            st.sidebar.info("Nothing new to process — those files are already indexed.")
 
 
# =========================================================================
# Document info
# =========================================================================
if st.session_state.doc_info:
    st.subheader("📚 Loaded documents")
    for info in st.session_state.doc_info:
        st.write(f"**{info['filename']}** — {info['pages']} page(s)/section(s), {info['chunks']} chunk(s)")
    st.write(f"Total chunks indexed: **{len(st.session_state.all_chunks)}**")
else:
    st.info("Upload documents or load them from Google Drive, then click 'Process documents' to get started.")
 
st.markdown("---")
 
 
# =========================================================================
# Ask a question
# =========================================================================
st.subheader("💬 Ask a question")
query = st.text_input("Your question about the documents")
ask_clicked = st.button("Ask")
 
if ask_clicked:
    if not st.session_state.all_chunks or st.session_state.faiss_index is None:
        st.warning("Please add and process at least one document first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            retrieved = hybrid_search(
                query, model, st.session_state.faiss_index,
                st.session_state.all_chunks, top_k=top_k, alpha=alpha,
            )
            answer = generate_answer(groq_client, query, retrieved)
 
        st.markdown("### Answer")
        st.write(answer)
 
        st.markdown("### Sources")
        for i, r in enumerate(retrieved, start=1):
            c = r["chunk"]
            page_info = f", page {c['page']}" if c["page"] else ""
            with st.expander(f"Source {i}: {c['filename']}{page_info} — score {r['score']:.2f}"):
                st.write(c["text"])
