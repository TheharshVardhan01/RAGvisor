import re
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def clean_text(text):
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line or "placeholder response" in line.lower():
            continue
        if re.match(r"^answer to '.*?':", line.lower()):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

    def load_and_split_url(url):
    """
    Load and split content from a URL into chunks with metadata.
    Args:
        url: The web page URL (str).
    Returns:
        List of tuples (chunk_text, metadata_dict)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
            tag.decompose()

        # Extract and clean text
        text = soup.get_text(separator="\n")
        cleaned_text = clean_text(text)

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_texts = splitter.split_text(cleaned_text)

        chunks = []
        for i, chunk in enumerate(split_texts):
            chunks.append((chunk, {"source": url, "chunk_id": f"{url}_chunk_{i}"}))

        return chunks
    except Exception as e:
        print(f"Error loading URL {url}: {e}")
        return []


def load_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return clean_text(text)

def load_text_file(file_path):
    return clean_text(file_path.read_text(encoding="utf-8"))

def load_and_split_pdfs(source, is_uploaded_files=False):
    chunks = []
    if is_uploaded_files:
        for uploaded_file in source:
            try:
                text = load_pdf_text(uploaded_file)
                split_texts = splitter.split_text(text)
                for i, chunk in enumerate(split_texts):
                    chunks.append((chunk, {"source": uploaded_file.name, "chunk_id": f"{uploaded_file.name}_{i}"}))
            except Exception as e:
                print(f"[ERROR] PDF load failed: {uploaded_file.name} — {e}")
    else:
        folder = Path(source)
        for file_path in folder.glob("*"):
            try:
                if file_path.suffix == ".pdf":
                    with open(file_path, "rb") as f:
                        text = load_pdf_text(f)
                elif file_path.suffix == ".txt":
                    text = load_text_file(file_path)
                else:
                    continue
                split_texts = splitter.split_text(text)
                for i, chunk in enumerate(split_texts):
                    chunks.append((chunk, {"source": file_path.name, "chunk_id": f"{file_path.name}_{i}"}))
            except Exception as e:
                print(f"[ERROR] File read failed: {file_path} — {e}")
    return chunks

def load_text_from_web(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        cleaned = clean_text(text)
        split_texts = splitter.split_text(cleaned)
        chunks = []
        for i, chunk in enumerate(split_texts):
            chunks.append((chunk, {"source": url, "chunk_id": f"{url}_{i}"}))
        return chunks
    except Exception as e:
        print(f"[ERROR] Web scraping failed: {url} — {e}")
        return []
