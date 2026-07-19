"""
Streamlit Web UI - PDF Embedding & Semantic Search.

Features:
1. Upload & Preview PDF - View PDF content and metadata
2. Export to JSON - Generate Gemini embeddings and export as JSON
3. Semantic Search - Search within exported JSON embedding files

Usage:
    streamlit run app.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import fitz  # PyMuPDF

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from embedding_service import GeminiEmbeddingService, GeminiGenerationService

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PDF Embedding Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main > div { padding: 0 1rem; }
    .card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .card h3 { margin-top: 0; color: #1f2937; }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-box h2 { margin: 0; font-size: 2rem; }
    .metric-box p { margin: 0; opacity: 0.9; font-size: 0.85rem; }
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .result-card .score { font-size: 1.3rem; font-weight: 700; color: #667eea; }
    .result-card .meta { font-size: 0.8rem; color: #6b7280; }
    .result-card .text { margin-top: 0.5rem; color: #374151; line-height: 1.5; }
    .stFileUploader > div {
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton button { border-radius: 8px; font-weight: 600; padding: 0.5rem 2rem; }
    .stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
    .info-box {
        background: #eff6ff; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #3b82f6; margin: 1rem 0;
    }
    .success-box {
        background: #f0fdf4; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #22c55e; margin: 1rem 0;
    }
    .warn-box {
        background: #fffbeb; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #f59e0b; margin: 1rem 0;
    }
    .error-box {
        background: #fef2f2; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #ef4444; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Utility Functions ───────────────────────────────────────────────────────

# Import reusable functions from existing modules
import search_json as sj
cosine_similarity = sj.cosine_similarity


def extract_text_from_bytes(file_bytes: bytes) -> Tuple[str, int, List[Tuple[int, str]]]:
    """Extract text from PDF bytes. Returns (full_text, page_count, pages_list)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = doc.page_count
    full_text = ""
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            pages.append((page_num, text))
            full_text += f"\n\n--- Page {page_num} ---\n\n{text}"
    doc.close()
    return full_text.strip(), page_count, pages


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        for delimiter in ["\n\n", "\n", ". ", " "]:
            last_boundary = text.rfind(delimiter, start, end)
            if last_boundary > start + chunk_size // 2:
                end = last_boundary + len(delimiter)
                break
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    return chunks


def estimate_page_for_chunk(chunk: str, pages: List[Tuple[int, str]], full_text: str) -> int:
    """Estimate which page a chunk belongs to."""
    char_pos = full_text.find(chunk[:50])
    if char_pos < 0:
        return 0
    cumulative = 0
    for page_num, page_text in pages:
        cumulative += len(page_text) + 2
        if char_pos < cumulative:
            return page_num
    return 0


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/google-logo.png", width=50)
    st.markdown("## PDF Embedding Studio")
    st.markdown("---")

    # API Key status
    has_api_key = bool(Config.GEMINI_API_KEY)
    if has_api_key:
        key_masked = Config.GEMINI_API_KEY[:8] + "..." + Config.GEMINI_API_KEY[-4:]
        st.markdown(
            f'<div class="success-box">🔑 **API Key:** `{key_masked}`</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="error-box">❌ **API Key tidak ditemukan**<br>'
            'Setel di file `.env`</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Settings")
    st.markdown(f"- **Model:** `{Config.EMBEDDING_MODEL}`")
    st.markdown(f"- **Dimensi:** `{Config.EMBEDDING_DIMENSIONALITY}`")
    st.markdown(f"- **Chunk Size:** `{Config.CHUNK_SIZE}`")
    st.markdown(f"- **Chunk Overlap:** `{Config.CHUNK_OVERLAP}`")
    st.markdown("---")
    st.markdown("Built with ❤️ using **Gemini Embedding** + **Streamlit**")


# ─── Session State Initialization ────────────────────────────────────────────

session_defaults = {
    "pdf_bytes": None,
    "pdf_name": None,
    "pdf_text": None,
    "pdf_pages": None,
    "pdf_page_count": 0,
    "pdf_extracted": False,
    "export_data": None,
    "export_filename": None,
    "processing": False,
    "chat_history": [],
}
for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─── Main Content ────────────────────────────────────────────────────────────

st.title("PDF Embedding Studio")
st.markdown("Upload PDF, buat embedding dengan **Gemini**, dan cari dengan **semantic search**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Upload & Preview PDF",
    "Export ke JSON",
    "Semantic Search",
    "Q&A Chat",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Upload & Preview PDF
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Upload PDF Document")

    uploaded_file = st.file_uploader(
        "Pilih file PDF",
        type=["pdf"],
        help="Upload dokumen PDF yang ingin di-embedding",
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        # Only re-extract if file is new or not yet extracted
        if (st.session_state.pdf_name != uploaded_file.name
                or not st.session_state.pdf_extracted):
            with st.spinner("Reading PDF..."):
                full_text, page_count, pages = extract_text_from_bytes(file_bytes)
                st.session_state.pdf_bytes = file_bytes
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_text = full_text
                st.session_state.pdf_pages = pages
                st.session_state.pdf_page_count = page_count
                st.session_state.pdf_extracted = True

        # Display metadata
        st.markdown("### Document Info")
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.markdown(
                f'<div class="metric-box"><h2>{st.session_state.pdf_page_count}</h2>'
                f"<p>Pages</p></div>",
                unsafe_allow_html=True
            )
        with meta_cols[1]:
            file_size_kb = len(file_bytes) / 1024
            st.markdown(
                f'<div class="metric-box"><h2>{file_size_kb:.1f}</h2><p>KB</p></div>',
                unsafe_allow_html=True
            )
        with meta_cols[2]:
            char_count = len(st.session_state.pdf_text)
            st.markdown(
                f'<div class="metric-box"><h2>{char_count:,}</h2><p>Characters</p></div>',
                unsafe_allow_html=True
            )
        with meta_cols[3]:
            st.markdown(
                f'<div class="metric-box"><h2>Chunked</h2><p>Approach</p></div>',
                unsafe_allow_html=True
            )

        # Preview text
        st.markdown("### Text Preview")
        preview = st.session_state.pdf_text[:3000]
        if len(st.session_state.pdf_text) > 3000:
            preview += "\n\n... (truncated, full text will be used during export)"
        st.text_area("Extracted Text", preview, height=300,
                      disabled=True, label_visibility="collapsed")

    else:
        st.info("Upload file PDF untuk memulai")
        st.markdown(
            '<div class="info-box">'
            "**Petunjuk:** Upload file PDF kamu, lalu pergi ke tab **Export ke JSON** "
            "untuk membuat embedding, atau tab **Semantic Search** untuk mencari dari "
            "file JSON yang sudah ada."
            "</div>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Export ke JSON
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Export PDF ke JSON Embedding")
    st.markdown(
        "Proses PDF dengan **Gemini Embedding** dan simpan hasilnya sebagai file JSON "
        "yang bisa dicari di tab **Semantic Search**."
    )

    if not has_api_key:
        st.markdown(
            '<div class="error-box">'
            "**API Key belum diatur!** Buat file `.env` dan isi `GEMINI_API_KEY=...`"
            "</div>",
            unsafe_allow_html=True
        )
        st.stop()

    if st.session_state.pdf_bytes is None:
        st.warning("Belum ada PDF yang diupload. Upload PDF dulu di tab **Upload & Preview PDF**.")
        st.stop()

    pdf_name = st.session_state.pdf_name
    page_count = st.session_state.pdf_page_count

    st.markdown(
            f'<div class="card">'
            f"<h3>{pdf_name}</h3>"
            f"<p><strong>Pages:</strong> {page_count} | "
            f"<strong>Chars:</strong> {len(st.session_state.pdf_text):,} | "
            f"<strong>Approach:</strong> Text Chunking (size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP})</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        export_btn = st.button(
            "Generate & Export",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing,
        )
    with col2:
        output_name = st.text_input(
            "Nama file output",
            value=pdf_name.replace(".pdf", "_embedding.json"),
            label_visibility="collapsed",
        )

    if export_btn:
        st.session_state.processing = True

        try:
            progress_bar = st.progress(0, text="Memulai...")
            status_text = st.empty()

            result = {
                "filename": pdf_name,
                "page_count": page_count,
                "approach": "text_extract_chunking",
                "embedding_model": Config.EMBEDDING_MODEL,
                "embedding_dimensions": Config.EMBEDDING_DIMENSIONALITY,
                "processed_at": datetime.now().isoformat(),
                "chunks": [],
            }

            embed_service = GeminiEmbeddingService()
            progress_bar.progress(10, text="Inisialisasi embedding service...")

            # Always use text extraction + chunking for granular search
            # (Even for short PDFs, chunking gives better search results)
            status_text.markdown("**Membagi teks menjadi chunk-chunk...**")
            full_text = st.session_state.pdf_text
            pages = st.session_state.pdf_pages

            chunks = chunk_text(full_text,
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP)

            status_text.markdown(f"**{len(chunks)} chunk dibuat. Mengirim ke Gemini...**")
            progress_bar.progress(30, text=f"Meng-embed {len(chunks)} chunks...")

            page_refs = [estimate_page_for_chunk(c, pages, full_text) for c in chunks]
            embeddings = embed_service.embed_texts(chunks)
            progress_bar.progress(70, text="Semua embedding berhasil!")

            for i, (chunk, emb, page) in enumerate(zip(chunks, embeddings, page_refs)):
                result["chunks"].append({
                    "chunk_index": i, "page": page,
                    "char_count": len(chunk),
                    "text": chunk, "embedding": emb,
                })
            progress_bar.progress(85, text="Menyusun data...")

            json_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
            json_size_kb = len(json_bytes) / 1024
            progress_bar.progress(100, text="Selesai!")

            st.markdown(
                f'<div class="success-box">'
                f"**Export berhasil!** - {len(result['chunks'])} chunks, {json_size_kb:.1f} KB"
                f"</div>",
                unsafe_allow_html=True
            )

            # Store in session state (single key for Tab 3)
            st.session_state.export_data = result
            st.session_state.export_filename = output_name

            st.download_button(
                label="Download JSON",
                data=json_bytes,
                file_name=output_name,
                mime="application/json",
                type="primary",
                use_container_width=True,
            )

            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            st.session_state.processing = False

    # Show recent export preview
    if st.session_state.get("export_data"):
        st.markdown("### Export Terakhir")
        d = st.session_state.export_data
        st.json({
            "filename": d["filename"],
            "page_count": d["page_count"],
            "approach": d["approach"],
            "chunks": len(d["chunks"]),
            "embedding_dimensions": d["embedding_dimensions"],
            "processed_at": d["processed_at"],
        })

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Semantic Search
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Semantic Search")
    st.markdown(
        "Cari dokumen menggunakan **semantic search** - pahami makna, bukan hanya kata kunci."
    )

    if not has_api_key:
        st.markdown(
            '<div class="error-box">'
            "**API Key belum diatur!** Buat file `.env` dan isi `GEMINI_API_KEY=...`"
            "</div>",
            unsafe_allow_html=True
        )
        st.stop()

    # Source selection
    st.markdown("#### Pilih Sumber Data")
    has_export = st.session_state.get("export_data") is not None

    col_src1, col_src2 = st.columns([1, 1])
    with col_src1:
        use_export = st.checkbox(
            "Gunakan hasil export terbaru",
            key="use_export_cb",
            disabled=not has_export,
        )
    with col_src2:
        uploaded_json = st.file_uploader(
            "Atau upload file JSON", type=["json"],
            key="json_uploader",
            help="Upload file JSON hasil export sebelumnya",
        )

    # Determine data source
    search_data = None
    search_name = None

    if use_export and has_export:
        search_data = st.session_state.export_data
        search_name = st.session_state.export_filename
        st.markdown(
            f'<div class="success-box">Menggunakan: **{search_name}**</div>',
            unsafe_allow_html=True
        )
    elif uploaded_json is not None:
        try:
            raw = json.loads(uploaded_json.read())
            if "chunks" not in raw or not raw["chunks"]:
                st.error("File JSON tidak valid: tidak ada chunks")
            else:
                required = {"chunk_index", "text", "embedding"}
                first = set(raw["chunks"][0].keys())
                if not required.issubset(first):
                    st.error(f"Format JSON tidak valid. Butuh keys: {required}")
                else:
                    search_data = raw
                    search_name = uploaded_json.name
                    st.markdown(
                        f'<div class="success-box">Loaded: **{search_name}** '
                        f'({len(search_data["chunks"])} chunks)</div>',
                        unsafe_allow_html=True
                    )
        except json.JSONDecodeError:
            st.error("File JSON tidak valid")

    if search_data is None:
        if not use_export and not uploaded_json:
            st.info("Pilih sumber data di atas untuk memulai pencarian.")
            st.markdown(
                '<div class="info-box">'
                "**Tips:** Export PDF dulu di tab **Export ke JSON**, "
                "lalu gunakan hasilnya untuk search di sini."
                "</div>",
                unsafe_allow_html=True
            )
        st.stop()

    # Dataset stats
    st.markdown("#### Dataset Info")
    cols = st.columns(4)
    with cols[0]: st.metric("Total Chunks", len(search_data["chunks"]))
    with cols[1]: st.metric("Embedding Dims", search_data.get("embedding_dimensions", "?"))
    with cols[2]: st.metric("Model", search_data.get("embedding_model", "?"))
    with cols[3]: st.metric("Pages", search_data.get("page_count", "?"))

    # Search
    st.markdown("#### Masukkan Pertanyaan")
    query = st.text_input(
        "Query",
        placeholder="Contoh: Apa keahlian utama?",
        label_visibility="collapsed",
    )

    col_k, col_btn = st.columns([1, 4])
    with col_k:
        top_k = st.number_input("Jumlah hasil", min_value=1, max_value=20, value=5)
    with col_btn:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("Mencari jawaban..."):
            try:
                embed_service = GeminiEmbeddingService()
                query_embedding = embed_service.embed_text(query)

                results = []
                for chunk in search_data["chunks"]:
                    score = cosine_similarity(query_embedding, chunk["embedding"])
                    results.append({
                        "score": score,
                        "chunk_index": chunk["chunk_index"],
                        "page": chunk.get("page"),
                        "char_count": chunk.get("char_count", 0),
                        "text": chunk["text"],
                        "source": search_data.get("filename", search_name),
                    })

                results.sort(key=lambda r: r["score"], reverse=True)
                results = results[:top_k]

                st.markdown(f"#### Search Results - {len(results)} ditemukan")

                for i, r in enumerate(results):
                    score_pct = r["score"] * 100
                    if score_pct >= 70:
                        emoji = "🟢"
                    elif score_pct >= 50:
                        emoji = "🟡"
                    else:
                        emoji = "🔵"

                    meta_parts = [f'Source: {r["source"]}']
                    if r["page"]:
                        meta_parts.append(f'Page: {r["page"]}')
                    meta_parts.append(f'Chunk #{r["chunk_index"]} ({r["char_count"]} chars)')
                    meta_str = " | ".join(meta_parts)

                    # Use expander as the container (Streamlit-native, no broken HTML)
                    with st.expander(
                        f"{emoji} #{i+1} — {score_pct:.1f}%  |  {meta_str}",
                        expanded=False,
                    ):
                        st.text_area(
                            label=f"Hasil {i+1}",
                            value=r["text"].strip(),
                            height=350,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"result_text_{i}",
                        )

            except Exception as e:
                st.error(f"Search gagal: {e}")

    elif search_btn and not query:
        st.warning("Masukkan pertanyaan terlebih dahulu.")

    st.markdown("---")
    st.caption("Powered by Google Gemini Embedding API")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Q&A Chat (RAG)
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Q&A Chat dengan Dokumen")
    st.markdown(
        "Tanya jawab langsung dengan isi PDF menggunakan **RAG** "
        "(Retrieval Augmented Generation). Gemini akan mencari chunk paling relevan "
        "lalu menjawab berdasarkan konteks dokumen kamu."
    )

    if not has_api_key:
        st.markdown(
            '<div class="error-box">'
            "**API Key belum diatur!** Buat file `.env` dan isi `GEMINI_API_KEY=...`"
            "</div>",
            unsafe_allow_html=True
        )
        st.stop()

    # Source selection (same pattern as Tab 3)
    st.markdown("#### Pilih Sumber Data")
    has_export = st.session_state.get("export_data") is not None

    col_src1, col_src2 = st.columns([1, 1])
    with col_src1:
        use_qa_export = st.checkbox(
            "Gunakan hasil export terbaru",
            key="use_qa_export_cb",
            disabled=not has_export,
        )
    with col_src2:
        qa_uploaded_json = st.file_uploader(
            "Atau upload file JSON", type=["json"],
            key="qa_json_uploader",
            help="Upload file JSON hasil export sebelumnya",
        )

    # Determine data source
    qa_data = None
    qa_name = None

    if use_qa_export and has_export:
        qa_data = st.session_state.export_data
        qa_name = st.session_state.export_filename
        st.markdown(
            f'<div class="success-box">Menggunakan: **{qa_name}**</div>',
            unsafe_allow_html=True
        )
    elif qa_uploaded_json is not None:
        try:
            raw = json.loads(qa_uploaded_json.read())
            if "chunks" not in raw or not raw["chunks"]:
                st.error("File JSON tidak valid: tidak ada chunks")
            else:
                qa_data = raw
                qa_name = qa_uploaded_json.name
                st.markdown(
                    f'<div class="success-box">Loaded: **{qa_name}** '
                    f'({len(qa_data["chunks"])} chunks)</div>',
                    unsafe_allow_html=True
                )
        except json.JSONDecodeError:
            st.error("File JSON tidak valid")

    if qa_data is None:
        st.info("Pilih sumber data di atas untuk mulai bertanya.")
        st.markdown(
            '<div class="info-box">'
            "**Tips:** Export PDF dulu di tab **Export ke JSON**, "
            "lalu gunakan hasilnya untuk tanya jawab di sini."
            "</div>",
            unsafe_allow_html=True
        )
        st.stop()

    # Dataset info
    st.markdown("#### Data Info")
    ci = st.columns(4)
    with ci[0]: st.metric("Total Chunks", len(qa_data["chunks"]))
    with ci[1]: st.metric("Model", qa_data.get("embedding_model", "?"))
    with ci[2]: st.metric("Generation Model", "gemini-2.0-flash")
    with ci[3]: st.metric("Pages", qa_data.get("page_count", "?"))

    # Chat interface
    st.markdown("---")
    st.markdown("#### Tanya Jawab")

    # Style chat messages
    st.markdown("""
    <style>
        .chat-msg-user {
            background: #e8f0fe;
            border-radius: 18px 18px 4px 18px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
            color: #1f2937;
        }
        .chat-msg-bot {
            background: #f3f4f6;
            border-radius: 18px 18px 18px 4px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
            max-width: 90%;
            color: #1f2937;
            line-height: 1.6;
            border-left: 3px solid #667eea;
        }
        .chat-msg-bot p { margin: 0.3rem 0; }
        .chat-msg-bot strong { color: #667eea; }
        .chat-timestamp {
            font-size: 0.7rem;
            color: #9ca3af;
            margin-top: 0.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Belum ada percakapan. Mulai dengan mengetik pertanyaan di bawah!")
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-msg-user">{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-msg-bot">{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )

    # Chat input
    col_q, col_btn = st.columns([4, 1])
    with col_q:
        qa_question = st.text_input(
            "Pertanyaan",
            placeholder="Contoh: Apa keahlian utama?",
            label_visibility="collapsed",
            key="qa_input",
        )
    with col_btn:
        ask_btn = st.button(
            "Tanya",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing,
        )

    # Number of chunks to retrieve
    st.caption(f"Akan mencari 3 chunk paling relevan sebagai konteks jawaban.")

    if ask_btn and qa_question:
        st.session_state.processing = True

        # Add user question to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": qa_question,
        })

        with st.spinner("Mencari konteks relevan dan menghasilkan jawaban..."):
            try:
                # Step 1: Embed the question
                embed_service = GeminiEmbeddingService()
                query_embedding = embed_service.embed_text(qa_question)

                # Step 2: Find top-3 relevant chunks
                scored = []
                for chunk in qa_data["chunks"]:
                    score = cosine_similarity(query_embedding, chunk["embedding"])
                    scored.append((score, chunk))
                scored.sort(key=lambda x: x[0], reverse=True)
                top_chunks = scored[:3]

                # Step 3: Ask Gemini with context
                gen_service = GeminiGenerationService()
                context_texts = [c["text"] for _, c in top_chunks]
                answer = gen_service.ask_with_context(
                    question=qa_question,
                    context_chunks=context_texts,
                    source_name=qa_name,
                )

                # Format answer with citations
                citations = []
                for score, chunk in top_chunks:
                    pct = score * 100
                    page_info = f" Halaman {chunk['page']}" if chunk.get("page") else ""
                    citations.append(
                        f"[Dokumen {len(citations)+1}] Relevansi: {pct:.0f}%"
                        f"{page_info} | Chunk #{chunk['chunk_index']}"
                    )

                formatted_answer = answer + "\n\n---\n" + "\n".join(citations)

                # Add answer to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": formatted_answer,
                })

                # Clear input
                st.session_state.qa_input = ""

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                st.session_state.processing = False

        # Rerun to show new messages
        st.rerun()

    elif ask_btn and not qa_question:
        st.warning("Masukkan pertanyaan terlebih dahulu.")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Hapus percakapan", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.caption("Powered by Google Gemini Embedding + Generation API")
