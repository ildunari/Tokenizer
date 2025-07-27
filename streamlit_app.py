"""
Token Counting Web Application - Redesigned
===========================================

A modern, compact Streamlit web application for token counting with improved aesthetics.
"""

import contextlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image
import PyPDF2

# Optional imports
try:
    import cv2
except Exception:
    cv2 = None

try:
    import wave
except Exception:
    wave = None

try:
    import requests
except Exception:
    requests = None


# Apply custom CSS for modern design
def apply_custom_css():
    st.markdown("""
    <style>
    /* Modern color scheme and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Compact header */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.7rem !important;
    }
    
    h4 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #4b5563 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-size: 0.9rem !important;
        border-radius: 8px !important;
        border: 2px solid #e5e7eb !important;
        transition: border-color 0.2s;
        resize: vertical;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        padding: 0.6rem !important;
        margin: 0.3rem 0 !important;
        font-size: 0.875rem !important;
    }
    
    [data-testid="stFileUploader"] {
        background: #f9fafb;
        border: 2px dashed #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        transition: border-color 0.2s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        font-size: 0.95rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important;
    }
    
    /* Selectboxes and multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        font-size: 0.9rem !important;
        min-height: 40px !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox label,
    .stMultiSelect label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    [data-testid="metric-container"] > div > div > div > div:first-child {
        font-size: 0.8rem !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="metric-container"] > div > div > div > div:last-child {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    
    /* Results table */
    .dataframe {
        font-size: 0.875rem !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: #f3f4f6 !important;
        font-weight: 600 !important;
        padding: 0.8rem !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        color: #374151 !important;
        border-bottom: 2px solid #e5e7eb !important;
    }
    
    .dataframe td {
        padding: 0.7rem 0.8rem !important;
        font-size: 0.85rem !important;
        border-bottom: 1px solid #f3f4f6 !important;
        color: #374151 !important;
    }
    
    /* Info cards */
    .info-card {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        line-height: 1.5;
        color: #4c51bf;
    }
    
    /* Warning and info messages */
    .stAlert {
        padding: 0.8rem !important;
        font-size: 0.875rem !important;
        border-radius: 8px !important;
    }
    
    /* Hide unnecessary elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Column gap adjustment */
    .row-widget.stHorizontal > div {
        gap: 1.5rem;
    }
    
    /* Results container */
    [data-testid="column"]:last-child {
        background: #fafbfc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    
    /* Help text styling */
    .stHelp {
        font-size: 0.8rem !important;
        color: #6b7280 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        [data-testid="column"]:last-child {
            margin-top: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


@dataclass
class TokenCounts:
    """Container for counting results for a single input."""
    name: str
    words: int
    characters: int
    openai_tokens: Optional[int] = None
    gemini_tokens: Optional[int] = None
    anthropic_tokens: Optional[int] = None


def get_context7_docs(library_name: str) -> Optional[str]:
    """Fetch up‑to‑date documentation for a library using Context7 MCP."""
    if requests is None:
        return None
    mcp_url = "https://mcp.context7.com/mcp"
    try:
        resolve_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resolve-library-id",
            "params": {"library_name": library_name},
        }
        with requests.post(
            mcp_url,
            json=resolve_payload,
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"},
            stream=True,
            timeout=10,
        ) as resp:
            library_id: Optional[str] = None
            for line_bytes in resp.iter_lines():
                if not line_bytes:
                    continue
                line = line_bytes.decode("utf-8", errors="ignore")
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        data_json = json.loads(data_str)
                        result = data_json.get("result")
                        if isinstance(result, list) and result:
                            library_id = result[0]
                            break
                    except Exception:
                        continue
        if not library_id:
            return None
        docs_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "get-library-docs",
            "params": {"context7_compatible_library_id": library_id, "tokens": 8000},
        }
        resp2 = requests.post(
            mcp_url,
            json=docs_payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        data = resp2.json()
        return data.get("result")
    except Exception:
        return None


def count_words(text: str) -> int:
    """Count the number of non‑whitespace token sequences (words)."""
    tokens = re.findall(r"\S+", text or "")
    return len(tokens)


def count_characters(text: str) -> int:
    """Count the number of characters in the provided string."""
    return len(text or "")


def approximate_token_count(text: str) -> int:
    """Approximate token count by assuming one token per ~4 characters."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens_openai(text: str, model: str) -> int:
    """Count tokens for OpenAI models using tiktoken when available."""
    try:
        import tiktoken
    except Exception:
        return approximate_token_count(text)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # Use the latest encoding for newer models
        try:
            if "gpt-4o" in model:
                enc = tiktoken.get_encoding("o200k_base")
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except:
            return approximate_token_count(text)
    try:
        return len(enc.encode(text))
    except Exception:
        return approximate_token_count(text)


def count_tokens_gemini_text(text: str, model: str) -> int:
    """Count tokens for Google Gemini models using Vertex AI when available."""
    try:
        from vertexai.preview.language_models import GenerativeModel

        # Handle model name variations
        if model.startswith("gemini-"):
            # Use the model name as-is for newer models
            gm = GenerativeModel(model)
        else:
            # Fallback for legacy names
            gm = GenerativeModel("gemini-1.5-flash")

        request = [{"role": "user", "parts": [text]}]
        response = gm.count_tokens(request)
        return int(response.total_tokens)
    except Exception:
        return approximate_token_count(text)


def count_tokens_anthropic(text: str, model: str, api_key: Optional[str]) -> int:
    """Count tokens for Anthropic Claude models via their API."""
    if not api_key:
        return approximate_token_count(text)
    try:
        import anthropic
    except Exception:
        return approximate_token_count(text)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": text}]
        response = client.messages.count_tokens(model=model, messages=messages)
        if isinstance(response, dict) and "input_tokens" in response:
            return int(response["input_tokens"])
        return int(getattr(response, "input_tokens"))
    except Exception:
        return approximate_token_count(text)


def gemini_tokens_image(image: Image.Image) -> int:
    """Compute Gemini token count for an image based on its dimensions."""
    width, height = image.size
    if width <= 384 and height <= 384:
        return 258
    tiles_x = (width + 767) // 768
    tiles_y = (height + 767) // 768
    return tiles_x * tiles_y * 258


def gemini_tokens_pdf_page() -> int:
    """Return the Gemini token cost for a PDF page."""
    return 258


def gemini_tokens_video(file_path: str) -> int:
    """Compute Gemini token count for a video using OpenCV to extract duration."""
    if cv2 is None:
        return 0
    cap = cv2.VideoCapture(file_path)
    if not cap or not cap.isOpened():
        return 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = (frame_count / fps) if fps > 0 else 0
    except Exception:
        duration = 0
    finally:
        cap.release()
    return int(duration * 263)


def gemini_tokens_audio(file_path: str) -> int:
    """Compute Gemini token count for an audio file (WAV only) via wave module."""
    if wave is None:
        return 0
    try:
        with contextlib.closing(wave.open(file_path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate) if rate > 0 else 0
        return int(duration * 32)
    except Exception:
        return 0


def extract_text_from_pdf(file: Any) -> str:
    """Extract all text from a PDF using PyPDF2."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_docx(file: Any) -> str:
    """Extract text from a DOCX file using python‑docx."""
    try:
        import docx
    except Exception:
        return ""
    try:
        document = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in document.paragraphs])
    except Exception:
        return ""


def process_uploaded_file(
    file: Any,
    selected_models: List[str],
    openai_model: Optional[str],
    gemini_model: Optional[str],
    anthropic_model: Optional[str],
    anthropic_key: Optional[str],
    temp_dir: str,
) -> TokenCounts:
    """Process an uploaded file and compute counts."""
    filename = file.name
    name_lower = filename.lower()
    ext = os.path.splitext(name_lower)[1]
    counts = TokenCounts(name=filename, words=0, characters=0)

    # Process text files
    if ext in {".txt", ".md", ".csv", ".log"}:
        try:
            text_bytes = file.read()
            try:
                text = text_bytes.decode("utf-8")
            except Exception:
                text = text_bytes.decode("latin-1", errors="ignore")
        except Exception:
            text = ""
        counts.words = count_words(text)
        counts.characters = count_characters(text)
        if "OpenAI" in selected_models and openai_model:
            counts.openai_tokens = count_tokens_openai(text, openai_model)
        if "Gemini" in selected_models and gemini_model:
            counts.gemini_tokens = count_tokens_gemini_text(text, gemini_model)
        if "Anthropic" in selected_models and anthropic_model:
            counts.anthropic_tokens = count_tokens_anthropic(
                text, anthropic_model, anthropic_key)
        return counts

    # Process PDF files
    if ext == ".pdf":
        pdf_text = extract_text_from_pdf(file)
        try:
            file.seek(0)
        except Exception:
            pass
        counts.words = count_words(pdf_text)
        counts.characters = count_characters(pdf_text)
        if "OpenAI" in selected_models and openai_model:
            counts.openai_tokens = count_tokens_openai(pdf_text, openai_model)
        if "Gemini" in selected_models and gemini_model:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                pages = len(pdf_reader.pages)
                counts.gemini_tokens = pages * gemini_tokens_pdf_page()
            except Exception:
                counts.gemini_tokens = None
        if "Anthropic" in selected_models and anthropic_model:
            counts.anthropic_tokens = count_tokens_anthropic(
                pdf_text, anthropic_model, anthropic_key)
        return counts

    # Process DOCX files
    if ext == ".docx":
        doc_text = extract_text_from_docx(file)
        counts.words = count_words(doc_text)
        counts.characters = count_characters(doc_text)
        if "OpenAI" in selected_models and openai_model:
            counts.openai_tokens = count_tokens_openai(doc_text, openai_model)
        if "Gemini" in selected_models and gemini_model:
            counts.gemini_tokens = count_tokens_gemini_text(
                doc_text, gemini_model)
        if "Anthropic" in selected_models and anthropic_model:
            counts.anthropic_tokens = count_tokens_anthropic(
                doc_text, anthropic_model, anthropic_key)
        return counts

    # Process images
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        try:
            image = Image.open(file)
            if "Gemini" in selected_models and gemini_model:
                counts.gemini_tokens = gemini_tokens_image(image)
        except Exception:
            pass
        return counts

    # Process video files
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}:
        if cv2 is not None and "Gemini" in selected_models and gemini_model:
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as tmp:
                tmp.write(file.read())
            counts.gemini_tokens = gemini_tokens_video(temp_path)
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return counts

    # Process audio files
    if ext in {".wav"}:
        if wave is not None and "Gemini" in selected_models and gemini_model:
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as tmp:
                tmp.write(file.read())
            counts.gemini_tokens = gemini_tokens_audio(temp_path)
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return counts

    return counts


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Token Counter Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Apply custom CSS
    apply_custom_css()

    # Header section
    st.markdown("# Token Counter Pro")
    st.markdown(
        '<p style="color: #6b7280; font-size: 0.95rem; margin-top: -0.5rem; margin-bottom: 1.5rem;">Analyze text, documents, and media with precision token counting</p>',
        unsafe_allow_html=True
    )

    # Main layout with text/file inputs on left, results on right
    input_col, results_col = st.columns([1.2, 1], gap="large")

    # Initialize variables that need to be accessible across columns
    count_button = False
    uploaded_files = None
    input_text = ""
    selected_models = []
    openai_model = None
    gemini_model = None
    anthropic_model = None
    anthropic_key = None

    with input_col:
        # Text input section
        st.markdown("### 📝 Text Input")
        input_text = st.text_area(
            "Paste your text here",
            placeholder="Enter or paste text to analyze...",
            height=150,
            label_visibility="collapsed"
        )

        # File upload section (visible at same time)
        st.markdown("### 📁 File Upload",
                    help="Upload multiple files to analyze together")
        uploaded_files = st.file_uploader(
            "Drop files or click to browse",
            type=["txt", "md", "csv", "log", "pdf", "docx", "png", "jpg",
                  "jpeg", "bmp", "gif", "wav", "mp4", "mov", "avi", "mkv",
                  "wmv", "flv"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.markdown(f'<div class="info-card">📎 {len(uploaded_files)} file(s) uploaded</div>',
                        unsafe_allow_html=True)

        # Settings section below inputs
        st.markdown("### ⚙️ Token Counter Settings")
        settings_container = st.container()

        with settings_container:
            # Model selection
            selected_models = st.multiselect(
                "Select tokenizers to use",
                ["OpenAI", "Gemini", "Anthropic"],
                default=["OpenAI", "Gemini"],
                label_visibility="visible"
            )

            # Model-specific settings
            openai_model = None
            gemini_model = None
            anthropic_model = None
            anthropic_key = None

            if selected_models:
                # Create columns for model dropdowns
                if len(selected_models) == 1:
                    col1, col2 = st.columns([1, 2])
                    model_cols = [col1]
                elif len(selected_models) == 2:
                    col1, col2 = st.columns(2)
                    model_cols = [col1, col2]
                else:
                    model_cols = st.columns(3)

                for idx, model in enumerate(selected_models):
                    with model_cols[idx]:
                        if model == "OpenAI":
                            openai_model = st.selectbox(
                                "OpenAI Model",
                                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
                                    "gpt-4", "gpt-3.5-turbo"],
                                index=0
                            )
                        elif model == "Gemini":
                            gemini_model = st.selectbox(
                                "Gemini Model",
                                ["gemini-2.5-pro-001", "gemini-2.5-flash-001",
                                 "gemini-2.0-flash-001", "gemini-1.5-pro-001",
                                 "gemini-1.5-flash-001"],
                                index=1
                            )
                        elif model == "Anthropic":
                            anthropic_model = st.selectbox(
                                "Anthropic Model",
                                ["claude-opus-4-20250514", "claude-sonnet-4-20250514",
                                 "claude-3.7-sonnet-20250219", "claude-3-opus-20240229",
                                 "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                                index=1
                            )

            # API key for Anthropic (if selected)
            if "Anthropic" in selected_models:
                anthropic_key = st.text_input(
                    "Anthropic API Key (optional)",
                    type="password",
                    placeholder="sk-ant-...",
                    help="Required for accurate Anthropic token counting"
                ) or None

            # Count button on the same row as settings
            button_col1, button_col2 = st.columns([2, 1])
            with button_col2:
                count_button = st.button(
                    "🚀 Count Tokens", use_container_width=True, type="primary")

    with results_col:
        # Results section - display when button is clicked
        if count_button:
            temp_dir = os.path.join(os.path.dirname(__file__), "_temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Process inputs - combine text and files
            results: List[TokenCounts] = []

            # Process text input
            if input_text:
                tc = TokenCounts(
                    name="Text Input",
                    words=count_words(input_text),
                    characters=count_characters(input_text),
                )
                if "OpenAI" in selected_models and openai_model:
                    tc.openai_tokens = count_tokens_openai(
                        input_text, openai_model)
                if "Gemini" in selected_models and gemini_model:
                    tc.gemini_tokens = count_tokens_gemini_text(
                        input_text, gemini_model)
                if "Anthropic" in selected_models and anthropic_model:
                    tc.anthropic_tokens = count_tokens_anthropic(
                        input_text, anthropic_model, anthropic_key)
                results.append(tc)

            # Process uploaded files
            for uf in uploaded_files or []:
                counts = process_uploaded_file(
                    uf, selected_models, openai_model, gemini_model,
                    anthropic_model, anthropic_key, temp_dir
                )
                results.append(counts)

            if results:
                st.markdown("### 📊 Results")

                # Summary metrics
                total_words = sum(r.words for r in results)
                total_chars = sum(r.characters for r in results)

                # Combined totals section
                st.markdown("#### Combined Totals")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Words", f"{total_words:,}")
                with col2:
                    st.metric("Total Characters", f"{total_chars:,}")

                # Model-specific totals
                if selected_models:
                    model_metrics = st.columns(len(selected_models))

                    if "OpenAI" in selected_models:
                        total_openai = sum(
                            r.openai_tokens or 0 for r in results)
                        idx = selected_models.index("OpenAI")
                        with model_metrics[idx]:
                            st.metric("OpenAI Tokens", f"{total_openai:,}")

                    if "Gemini" in selected_models:
                        total_gemini = sum(
                            r.gemini_tokens or 0 for r in results)
                        idx = selected_models.index("Gemini")
                        with model_metrics[idx]:
                            st.metric("Gemini Tokens", f"{total_gemini:,}")

                    if "Anthropic" in selected_models:
                        total_anthropic = sum(
                            r.anthropic_tokens or 0 for r in results)
                        idx = selected_models.index("Anthropic")
                        with model_metrics[idx]:
                            st.metric("Anthropic Tokens",
                                      f"{total_anthropic:,}")

                # Detailed breakdown
                if len(results) > 1:
                    st.markdown("#### Breakdown by Source")

                    # Convert to DataFrame for better display
                    import pandas as pd

                    table_data = []
                    for r in results:
                        row = {
                            "Source": r.name[:30] + "..." if len(r.name) > 30 else r.name,
                            "Words": f"{r.words:,}",
                            "Chars": f"{r.characters:,}"
                        }
                        if "OpenAI" in selected_models:
                            row["OpenAI"] = f"{r.openai_tokens:,}" if r.openai_tokens else "—"
                        if "Gemini" in selected_models:
                            row["Gemini"] = f"{r.gemini_tokens:,}" if r.gemini_tokens else "—"
                        if "Anthropic" in selected_models:
                            row["Anthropic"] = f"{r.anthropic_tokens:,}" if r.anthropic_tokens else "—"
                        table_data.append(row)

                    df = pd.DataFrame(table_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=min(250, 40 + len(df) * 35)
                    )
            else:
                st.warning("⚠️ Please enter text or upload files to analyze")
        else:
            # Show placeholder when no results
            st.markdown("### 📊 Results")
            st.info(
                "Enter text and/or upload files, then click 'Count Tokens' to see results here")


if __name__ == "__main__":
    main()
