# AI Document Assistant

A simple Streamlit app that lets you upload documents (or load them from
Google Drive), turns them into searchable chunks, and answers questions
about them using hybrid search + a Groq LLM — grounded only in your
documents.

## Features

- **Upload** PDF, DOCX, TXT, and MD files (or load them from a public
  Google Drive file/folder link).
- **Extraction**: a separate function per file type. PDFs keep page
  numbers; other formats are treated as a single section.
- **Chunking**: text is split into overlapping word chunks, each tagged
  with its source filename and page number.
- **Embeddings**: chunks are embedded once with `sentence-transformers`
  (`all-MiniLM-L6-v2`) and kept in memory (`st.session_state`) so they are
  never recomputed while you ask more questions.
- **Search**: a FAISS index for semantic search, a simple keyword scorer,
  and a hybrid function that blends both (adjustable with a slider).
- **Answers**: retrieved chunks are sent to Groq's LLM, which is instructed
  to answer only from that context and say when something isn't covered.
- **Sources**: every answer is followed by the exact chunks used, with
  filename and page number.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys to Streamlit secrets. Create a file at
   `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"

   # Optional: only needed if you want to load an entire Google Drive
   # folder (a single public file link works without this).
   GOOGLE_API_KEY = "your-google-api-key"
   ```

   Never put these keys directly in `app.py`.

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## Using Google Drive

- **Single file link** (e.g. `https://drive.google.com/file/d/FILE_ID/view`):
  works out of the box as long as the file's sharing setting is
  "Anyone with the link can view."
- **Folder link** (e.g. `https://drive.google.com/drive/folders/FOLDER_ID`):
  requires `GOOGLE_API_KEY` (a Google Cloud API key with the Drive API
  enabled) and the folder must be publicly shared.

## How it works, in short

1. `extract_*` functions pull raw text (and page numbers, for PDFs) out of
   each file.
2. `chunk_text` splits that text into overlapping ~180-word pieces, each
   carrying its filename/page metadata.
3. `embed_chunks` turns every chunk into a vector once; `build_faiss_index`
   stores those vectors for fast lookup.
4. When you ask a question, `hybrid_search` blends FAISS semantic
   similarity with a keyword-overlap score to pick the best chunks.
5. `generate_answer` sends those chunks plus your question to Groq, asking
   it to answer strictly from the provided context.

## Notes & limitations

- Already-processed files aren't re-embedded — the app tracks them by
  filename + size, so clicking "Process documents" again only embeds new
  files.
- DOCX/TXT/MD files don't have real page numbers, so their chunks show no
  page — this is expected.
- Very large Drive files may need an extra confirmation step that Google
  adds automatically for its virus scan; the app handles the common case
  but extremely large files may still fail to download.
