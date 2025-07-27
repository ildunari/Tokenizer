"""
Token Counting Web Application
================================

This Streamlit web application allows users to paste text or upload files and
receive counts for words, characters and tokens.  In addition to simple counts
the application can approximate token counts for three different families of
language models—OpenAI, Google Gemini and Anthropic Claude.  Whenever
possible the application attempts to use official tokenizers.  If a tokenizer
library is not available in the execution environment, a fallback heuristic
(`len(text) // 4`) is used.  The app also handles basic multimedia file
types supported by Gemini’s token accounting rules: images are converted to
their dimensions to determine the number of 768×768 tiles needed, video
duration is extracted via OpenCV to estimate tokens at 263 tokens per second
and WAV audio duration is measured to estimate tokens at 32 tokens per
second.  Other audio formats are ignored unless the environment contains
additional decoding libraries.

The code also includes optional integration with the Context7 Model Context
Protocol (MCP) service.  If the environment can reach the Context7 MCP
endpoint it will resolve library names and fetch up‑to‑date documentation for
the libraries used in this application.  These docs are not displayed to the
user but can be inspected via the return value of `get_context7_docs()`.

To run this application locally you need to install the required
dependencies listed in the accompanying `requirements.txt` file.  Once
installed, execute the app with:

```
streamlit run app.py
```

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

# Optional imports.  These may not be available in all environments.  The
# application gracefully falls back when they are missing.
try:
    import cv2  # type: ignore[import]
except Exception:
    cv2 = None  # type: ignore[assignment]

try:
    import wave  # standard library, used for WAV audio duration
except Exception:
    wave = None  # type: ignore[assignment]

# Context7 integration uses the requests library.  If this package is not
# installed the function will silently return None.
try:
    import requests  # type: ignore[import]
except Exception:
    requests = None  # type: ignore[assignment]


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
    """Fetch up‑to‑date documentation for a library using Context7 MCP.

    The MCP server exposes two methods: ``resolve-library-id`` and
    ``get-library-docs``.  This helper first resolves a Context7‑compatible
    library identifier from a human friendly name then fetches the
    corresponding documentation.  When an error occurs or the ``requests``
    package is unavailable it returns None.  Documentation retrieval uses
    JSON‑RPC over HTTP.  See Context7 docs for more details.

    Parameters
    ----------
    library_name: str
        The name of the library to resolve (e.g. ``tiktoken`` or
        ``google-cloud-aiplatform``).

    Returns
    -------
    Optional[str]
        A markdown string containing the library documentation or None if
        retrieval fails.
    """
    if requests is None:
        return None
    mcp_url = "https://mcp.context7.com/mcp"
    try:
        # Resolve the library ID.  We need to use Server Sent Events (SSE)
        # transport because the default response uses SSE frames.  Setting
        # Accept to ``text/event-stream`` tells the server to stream events.
        resolve_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resolve-library-id",
            "params": {"library_name": library_name},
        }
        with requests.post(
            mcp_url,
            json=resolve_payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            stream=True,
            timeout=10,
        ) as resp:
            library_id: Optional[str] = None
            for line_bytes in resp.iter_lines():
                if not line_bytes:
                    continue
                line = line_bytes.decode("utf-8", errors="ignore")
                if line.startswith("data:"):
                    # Each SSE event line begins with ``data:`` followed by a JSON string.
                    data_str = line[len("data:"):].strip()
                    try:
                        data_json = json.loads(data_str)
                        result = data_json.get("result")
                        if isinstance(result, list) and result:
                            # Use the first ID in the result list.
                            library_id = result[0]
                            break
                    except Exception:
                        continue
        if not library_id:
            return None
        # Now fetch the documentation.  The ``get-library-docs`` method
        # expects the resolved ID and optionally a topic and token limit.
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
        # In case of any network or JSON error return None.  The caller
        # handles the absence of documentation gracefully.
        return None


def count_words(text: str) -> int:
    """Count the number of non‑whitespace token sequences (words).

    Uses a regular expression to split on one or more whitespace characters.
    """
    tokens = re.findall(r"\S+", text or "")
    return len(tokens)


def count_characters(text: str) -> int:
    """Count the number of characters in the provided string."""
    return len(text or "")


def approximate_token_count(text: str) -> int:
    """Approximate token count by assuming one token per ~4 characters.

    This heuristic is based on observations that GPT and Gemini tokens
    average roughly four characters.  It acts as a fallback when a real
    tokenizer library (e.g. tiktoken, sentencepiece or anthropic) is not
    available.  Always returns at least one token for non‑empty input.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens_openai(text: str, model: str) -> int:
    """Count tokens for OpenAI models using tiktoken when available.

    If the tiktoken package cannot be imported or the model’s encoding is
    unknown the function falls back to the approximate token heuristic.

    Parameters
    ----------
    text: str
        The text to tokenize.
    model: str
        An OpenAI model name such as ``gpt-4o`` or ``gpt-3.5-turbo``.  The
        ``tiktoken.encoding_for_model`` function automatically maps model
        names to encodings.

    Returns
    -------
    int
        The number of tokens in the given text.
    """
    try:
        import tiktoken  # type: ignore[import]
    except Exception:
        return approximate_token_count(text)
    try:
        # Use model specific encoding when available.
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to a common encoding if the model is unknown.
        enc = tiktoken.get_encoding("cl100k_base")
    try:
        return len(enc.encode(text))
    except Exception:
        return approximate_token_count(text)


def count_tokens_gemini_text(text: str, model: str) -> int:
    """Count tokens for Google Gemini models using Vertex AI when available.

    The Vertex AI SDK supports local tokenization starting from version
    1.57.0 via the preview generative models.  If the SDK cannot be
    imported or the model does not support token counting this function
    falls back to the approximate heuristic.  Because the generative
    models API may change, using Context7 to fetch up‑to‑date docs is
    advised when deploying this app in production.
    """
    try:
        # Import inside the function to avoid import errors when the package
        # is unavailable.
        from vertexai.preview.language_models import GenerativeModel  # type: ignore[import]

        gm = GenerativeModel(model)
        # Compose a Gemini chat message.  A single user message can be
        # represented as a list of dictionaries with ``role`` and ``parts``.
        request = [{"role": "user", "parts": [text]}]
        response = gm.count_tokens(request)
        # The response object has a ``total_tokens`` attribute.
        return int(response.total_tokens)
    except Exception:
        return approximate_token_count(text)


def count_tokens_anthropic(text: str, model: str, api_key: Optional[str]) -> int:
    """Count tokens for Anthropic Claude models via their API.

    When an API key is provided the function calls the official ``count_tokens``
    endpoint using the Anthropic Python client.  If no key is supplied or the
    client cannot be imported the function falls back to the approximate
    heuristic.  Note that calling the Anthropic API will consume network
    requests and should be used sparingly.
    """
    if not api_key:
        return approximate_token_count(text)
    try:
        import anthropic  # type: ignore[import]
    except Exception:
        return approximate_token_count(text)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # The Claude API expects messages as a list of dicts.  Only the
        # ``user`` role is used here.
        messages = [{"role": "user", "content": text}]
        response = client.messages.count_tokens(model=model, messages=messages)
        # The API returns a dict with an ``input_tokens`` field.
        if isinstance(response, dict) and "input_tokens" in response:
            return int(response["input_tokens"])
        # Some versions return an object with attributes.
        return int(getattr(response, "input_tokens"))
    except Exception:
        return approximate_token_count(text)


def gemini_tokens_image(image: Image.Image) -> int:
    """Compute Gemini token count for an image based on its dimensions.

    According to Google’s token accounting rules, images up to 384×384
    contribute 258 tokens.  Larger images are tiled into 768×768 patches
    and each tile counts as 258 tokens.  PDFs are treated like images at
    the page level and processed separately.  See Google’s docs for
    details【617856474601913†L532-L548】.
    """
    width, height = image.size
    # Small images count as a single tile of 258 tokens.
    if width <= 384 and height <= 384:
        return 258
    # Larger images are divided into 768×768 tiles.  Compute the number
    # of tiles in each dimension by rounding up.
    tiles_x = (width + 767) // 768
    tiles_y = (height + 767) // 768
    return tiles_x * tiles_y * 258


def gemini_tokens_pdf_page() -> int:
    """Return the Gemini token cost for a PDF page.

    Google treats PDF pages similarly to images and assigns 258 tokens per
    page for billing purposes【617856474601913†L532-L553】.  This constant
    simplifies PDF token counting without requiring image conversion.
    """
    return 258


def gemini_tokens_video(file_path: str) -> int:
    """Compute Gemini token count for a video using OpenCV to extract duration.

    Videos contribute 263 tokens per second【617856474601913†L542-L548】.  If
    OpenCV is not available or the file cannot be opened, returns zero.
    """
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
    """Compute Gemini token count for an audio file (WAV only) via wave module.

    Audio contributes 32 tokens per second【617856474601913†L542-L548】.  Only
    WAV format is supported because Python’s standard ``wave`` module can
    read WAV files.  Other audio formats return zero tokens unless a
    decoding library is available.
    """
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
    """Extract all text from a PDF using PyPDF2.

    Parameters
    ----------
    file: Any
        A binary file object returned from Streamlit’s file uploader.

    Returns
    -------
    str
        Concatenated text from all pages of the PDF.
    """
    try:
        # PyPDF2 can accept a file‑like object directly.
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_docx(file: Any) -> str:
    """Extract text from a DOCX file using python‑docx.

    Returns an empty string if python‑docx is unavailable or parsing fails.
    """
    try:
        import docx  # type: ignore[import]
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
    """Process an uploaded file and compute counts.

    Based on the file extension this function decides how to extract text or
    dimensional information.  Textual files (TXT, PDF, DOCX) produce both
    word and character counts as well as tokens for each selected model.  Image
    files produce only Gemini tokens because other models do not currently
    accept images via purely offline tokenization.  Video and audio files
    similarly produce Gemini token counts.  Unsupported file types yield
    zero counts for words and characters.

    Parameters
    ----------
    file: Any
        The uploaded file object from Streamlit.
    selected_models: List[str]
        A list of tokenizers selected by the user.
    openai_model: Optional[str]
        The OpenAI model name to use.
    gemini_model: Optional[str]
        The Gemini model name to use.
    anthropic_model: Optional[str]
        The Anthropic model name to use.
    anthropic_key: Optional[str]
        The user‑supplied API key for Anthropic.
    temp_dir: str
        Directory on disk where temporary files can be written.

    Returns
    -------
    TokenCounts
        A record with counts for the file.
    """
    filename = file.name
    name_lower = filename.lower()
    ext = os.path.splitext(name_lower)[1]
    # Initialize counts with zeros.  We fill fields as appropriate below.
    counts = TokenCounts(name=filename, words=0, characters=0)
    # Process text files.
    if ext in {".txt", ".md", ".csv", ".log"}:
        try:
            text_bytes = file.read()
            # Attempt to decode using UTF‑8.  Fallback to Latin‑1 if decoding fails.
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
            counts.anthropic_tokens = count_tokens_anthropic(text, anthropic_model, anthropic_key)
        return counts
    # Process PDF files.
    if ext == ".pdf":
        pdf_text = extract_text_from_pdf(file)
        counts.words = count_words(pdf_text)
        counts.characters = count_characters(pdf_text)
        if "OpenAI" in selected_models and openai_model:
            counts.openai_tokens = count_tokens_openai(pdf_text, openai_model)
        if "Gemini" in selected_models and gemini_model:
            # Each PDF page contributes 258 tokens.  We measure the number of
            # pages from the PDF reader.
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                pages = len(pdf_reader.pages)
                counts.gemini_tokens = pages * gemini_tokens_pdf_page()
            except Exception:
                counts.gemini_tokens = None
        if "Anthropic" in selected_models and anthropic_model:
            counts.anthropic_tokens = count_tokens_anthropic(pdf_text, anthropic_model, anthropic_key)
        return counts
    # Process DOCX files.
    if ext == ".docx":
        doc_text = extract_text_from_docx(file)
        counts.words = count_words(doc_text)
        counts.characters = count_characters(doc_text)
        if "OpenAI" in selected_models and openai_model:
            counts.openai_tokens = count_tokens_openai(doc_text, openai_model)
        if "Gemini" in selected_models and gemini_model:
            counts.gemini_tokens = count_tokens_gemini_text(doc_text, gemini_model)
        if "Anthropic" in selected_models and anthropic_model:
            counts.anthropic_tokens = count_tokens_anthropic(doc_text, anthropic_model, anthropic_key)
        return counts
    # Process images.
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        try:
            image = Image.open(file)
            if "Gemini" in selected_models and gemini_model:
                counts.gemini_tokens = gemini_tokens_image(image)
        except Exception:
            pass
        # Images have no text; words and characters remain zero.  Other models
        # cannot compute tokens offline without OCR so we leave their counts
        # as None.
        return counts
    # Process video files.  Save to a temporary file so OpenCV can access it.
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}:
        if cv2 is not None and "Gemini" in selected_models and gemini_model:
            # Write the bytes to disk.
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as tmp:
                tmp.write(file.read())
            counts.gemini_tokens = gemini_tokens_video(temp_path)
            # Clean up the temp file after reading.
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return counts
    # Process audio files.  We only support WAV due to the wave module.
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
    # If file type is unknown, we leave counts at zero/None.
    return counts


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Token Counter", page_icon="📏", layout="wide")
    st.title("Token Counter and Analyzer")
    st.markdown(
        """
        **Overview**
        
        This application allows you to paste text or upload files and see
        statistics on their content.  It computes:
        
        - **Words**: number of whitespace‑separated sequences
        - **Characters**: total number of characters
        - **Tokens**: approximate counts for OpenAI GPT, Google Gemini and Anthropic Claude models
        
        When the necessary libraries are installed, the counts use the same
        tokenizer implementations as the respective platforms; otherwise a simple
        heuristic (four characters per token) is used.  For Gemini, images and
        multimedia files are supported and follow Google’s documented token
        accounting rules【617856474601913†L532-L548】.  Token counts for Anthropic
        models can use your API key if supplied; if no key is provided a
        heuristic is used.  Use the control panel below to configure the
        tokenizers.
        """
    )

    # Input text area
    input_text = st.text_area("Paste text here", height=200)
    uploaded_files = st.file_uploader(
        "Upload files (text, PDF, DOCX, images, audio, video)",
        type=[
            "txt",
            "md",
            "csv",
            "log",
            "pdf",
            "docx",
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "gif",
            "wav",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "wmv",
            "flv",
        ],
        accept_multiple_files=True,
    )

    st.markdown("### Tokenizer Settings")
    selected_models = st.multiselect(
        "Select tokenizers to use", ["OpenAI", "Gemini", "Anthropic"], default=["OpenAI", "Gemini"]
    )
    # Options for each model family
    openai_model = None
    gemini_model = None
    anthropic_model = None
    anthropic_key: Optional[str] = None
    if "OpenAI" in selected_models:
        openai_model = st.selectbox(
            "OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Model name passed to tiktoken for encoding."
        )
    if "Gemini" in selected_models:
        gemini_model = st.selectbox(
            "Gemini model", [
                "gemini-1.5-pro-001",
                "gemini-1.5-flash-001",
                "gemini-pro",  # alias for older names
                "gemini-1.0-pro",
            ],
            index=0,
            help="Model name passed to Vertex AI tokenization."
        )
    if "Anthropic" in selected_models:
        anthropic_model = st.selectbox(
            "Anthropic model", [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240229",
            ],
            index=1,
            help="Model name used for Anthropic Claude token counting."
        )
        anthropic_key = st.text_input(
            "Anthropic API key (optional)", value="", type="password", help="Used for exact Claude token counting."
        ) or None

    # Temporary directory for media files
    temp_dir = os.path.join(os.path.dirname(__file__), "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    if st.button("Count Tokens"):
        # Optionally fetch up‑to‑date docs via Context7.  This is a
        # demonstration of how the tool can be integrated; the docs are not
        # displayed but could be used for validation or debugging.
        with st.spinner("Resolving documentation via Context7..."):
            _ = get_context7_docs("tiktoken")
            _ = get_context7_docs("google-cloud-aiplatform")
            _ = get_context7_docs("anthropic")
        # Process text input
        results: List[TokenCounts] = []
        if input_text:
            tc = TokenCounts(
                name="Input Text",
                words=count_words(input_text),
                characters=count_characters(input_text),
            )
            if "OpenAI" in selected_models and openai_model:
                tc.openai_tokens = count_tokens_openai(input_text, openai_model)
            if "Gemini" in selected_models and gemini_model:
                tc.gemini_tokens = count_tokens_gemini_text(input_text, gemini_model)
            if "Anthropic" in selected_models and anthropic_model:
                tc.anthropic_tokens = count_tokens_anthropic(input_text, anthropic_model, anthropic_key)
            results.append(tc)
        # Process uploaded files
        for uf in uploaded_files or []:
            counts = process_uploaded_file(
                uf,
                selected_models,
                openai_model,
                gemini_model,
                anthropic_model,
                anthropic_key,
                temp_dir,
            )
            results.append(counts)
        # Display results as a table
        if results:
            # Convert list of dataclasses to list of dicts for Streamlit table.
            table: List[Dict[str, Any]] = []
            for r in results:
                table.append(
                    {
                        "Name": r.name,
                        "Words": r.words,
                        "Characters": r.characters,
                        "OpenAI Tokens": r.openai_tokens,
                        "Gemini Tokens": r.gemini_tokens,
                        "Anthropic Tokens": r.anthropic_tokens,
                    }
                )
            st.markdown("### Results")
            st.table(table)
        else:
            st.warning("Please enter text or upload at least one supported file.")


if __name__ == "__main__":
    main()
