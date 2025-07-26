import streamlit as st
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
from dotenv import load_dotenv
import hashlib
import re
import time
import bleach
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import OrderedDict

# Sample implementations of missing modules
def load_and_split_pdfs(pdf_input, is_uploaded_files=False):
    """Load and split PDFs into chunks."""
    try:
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        if is_uploaded_files:
            for file in pdf_input:
                reader = PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                chunks.extend(text_splitter.split_text(text))
        else:
            for file_path in Path(pdf_input).glob("*.pdf"):
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                chunks.extend(text_splitter.split_text(text))
        return chunks
    except Exception as e:
        st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Failed to load PDFs: {e}</div>', unsafe_allow_html=True)
        return []

def embed_and_store(chunks, persist_dir):
    """Embed chunks and store in ChromaDB."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)
        client = PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection("rag_pdf")
        # Generate unique IDs for each chunk
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": "unknown", "chunk_id": i} for i in range(len(chunks))],
            ids=ids  # Add the required ids parameter
        )
    except Exception as e:
        st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Failed to embed content: {e}</div>', unsafe_allow_html=True)
from groq import Groq

def generate_answer(query, context):
    """Generate an answer using Groq API."""
    try:
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is missing in environment variables.")
        
        client = Groq(api_key=groq_api_key)
        
        # Construct the prompt
        prompt = f"""You are an AI assistant tasked with answering questions based on provided context. Use the following context to answer the query concisely and accurately. If the context is insufficient, provide a general answer or indicate limitations.

        Query: {query}

        Context: {context}

        Answer:"""
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",  # Adjust model as needed (e.g., llama3-8b-8192)
            max_tokens=1000,  # Increase for longer responses
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {e}"
# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
required_env_vars = ["DEEPAI_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Environment variable {var} is missing. Please set it in your .env file.</div>', unsafe_allow_html=True)
        st.stop()

# ========== Page Config ==========
st.set_page_config(page_title="RAGvisor", layout="wide", page_icon="🧠")

# ========== Initialize State ==========
st.session_state.setdefault("qa_history", [])
st.session_state.setdefault("images", [])
st.session_state.setdefault("dark_mode", False)
st.session_state.setdefault("query_cache", OrderedDict())
st.session_state.setdefault("chat_history_visible", True)

# ========== Custom CSS ==========
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
        
        :root {
            --bg-color: #ffffff;
            --text-color: #1e293b;
            --primary-color: #2563eb;
            --secondary-color: #047857;
            --accent-color: #f59e0b;
            --chunk-bg: #f9fafb;
            --chunk-border: #f59e0b;
            --user-bg: #e0f2fe;
            --bot-bg: #e6f6f2;
            --bot-border: #047857;
            --input-bg: #f1f5f9;
            --button-bg: #2563eb;
            --button-text: #ffffff;
            --border-radius: 8px;
            --shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            --error-bg: #fee2e2;
            --error-text: #dc2626;
            --success-bg: #d1fae5;
            --success-text: #047857;
            --warning-bg: #fefcbf;
            --warning-text: #854d0e;
        }
        [data-theme="dark"] {
            --bg-color: #1e293b;
            --text-color: #e2e8f0;
            --primary-color: #60a5fa;
            --secondary-color: #34d399;
            --accent-color: #fbbf24;
            --chunk-bg: #334155;
            --chunk-border: #fbbf24;
            --user-bg: #475569;
            --bot-bg: #1e3a8a;
            --bot-border: #34d399;
            --input-bg: #2d3748;
            --button-bg: #60a5fa;
            --button-text: #ffffff;
            --error-bg: #451a1a;
            --error-text: #f87171;
            --success-bg: #1e3932;
            --success-text: #34d399;
            --warning-bg: #3f3b13;
            --warning-text: #facc15;
        }
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(45deg, var(--bg-color), #f0f4f8);
            color: var(--text-color);
            transition: all 0.3s ease;
            margin: 0;
            padding: 0;
            animation: backgroundShift 10s ease infinite;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 1rem;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
            animation: bounce 2s infinite;
        }
        .chunk-card, .image-card {
            background: var(--chunk-bg);
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid var(--chunk-border);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            transition: transform 0.2s ease;
        }
        .chunk-card:hover, .image-card:hover {
            transform: translateY(-2px);
        }
        .message.user, .message.bot {
            padding: 1rem;
            border-radius: var(--border-radius);
            margin: 0.5rem 0;
            box-shadow: var(--shadow);
            max-width: 80%;
            animation: fadeInBounce 0.5s ease-out;
        }
        .message.user {
            background-color: var(--user-bg);
            margin-left: auto;
        }
        .message.bot {
            background-color: var(--bot-bg);
            border-left: 5px solid var(--bot-border);
            margin-right: auto;
        }
        .chat-history-container {
            display: {display};
            transition: all 0.3s ease;
        }
        .toggle-chat {
            background-color: var(--button-bg);
            color: var(--button-text);
            border: none;
            border-radius: var(--border-radius);
            padding: 0.5rem 1rem;
            cursor: pointer;
            margin-bottom: 1rem;
            transition: background-color 0.2s ease;
        }
        .toggle-chat:hover {
            background-color: var(--secondary-color);
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: var(--input-bg) !important;
            color: var(--text-color) !important;
            border-radius: var(--border-radius);
            border: 1px solid var(--primary-color);
            padding: 0.75rem;
            font-size: 1rem;
        }
        .stButton > button {
            background-color: var(--button-bg) !important;
            color: var(--button-text) !important;
            border-radius: var(--border-radius);
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        .stButton > button:hover {
            background-color: var(--secondary-color) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .stButton > button::after {
            content: '';
            position: absolute;
            width: 0;
            height: 100%;
            top: 0;
            left: 0;
            background: rgba(255, 255, 255, 0.2);
            animation: pulse 1.5s infinite;
        }
        .stButton > button:hover::after {
            width: 100%;
        }
        .upload-dropzone {
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            background-color: var(--input-bg);
            margin-bottom: 1rem;
            transition: border-color 0.3s ease;
        }
        .upload-dropzone.dragover {
            border-color: var(--secondary-color);
            background-color: #e6f6f2;
        }
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .image-gallery img {
            max-width: 250px;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            transition: transform 0.2s ease;
            animation: slideIn 0.5s ease-out;
        }
        .image-gallery img:hover {
            transform: scale(1.05);
            cursor: zoom-in;
        }
        .stSpinner > div > div {
            border-color: var(--primary-color) transparent transparent transparent !important;
            animation: spin 1s linear infinite !important;
            width: 30px !important;
            height: 30px !important;
        }
        .custom-error {
            background-color: var(--error-bg);
            color: var(--error-text);
            padding: 0.75rem;
            border-radius: var(--border-radius);
            margin: 0.5rem 0;
            box-shadow: var(--shadow);
            animation: fadeIn 0.3s ease-out;
        }
        .custom-success {
            background-color: var(--success-bg);
            color: var(--success-text);
            padding: 0.75rem;
            border-radius: var(--border-radius);
            margin: 0.5rem 0;
            box-shadow: var(--shadow);
            animation: fadeIn 0.3s ease-out;
        }
        .custom-warning {
            background-color: var(--warning-bg);
            color: var(--warning-text);
            padding: 0.75rem;
            border-radius: var(--border-radius);
            margin: 0.5rem 0;
            box-shadow: var(--shadow);
            animation: fadeIn 0.3s ease-out;
        }
        @keyframes fadeInBounce {
            0% { opacity: 0; transform: translateY(20px); }
            50% { opacity: 0.5; transform: translateY(-10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
            0% { left: 0; width: 0; }
            20% { left: 0; width: 100%; }
            100% { left: 100%; width: 0; }
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        html {
            background-size: 200% 200%;
        }
        @media (max-width: 768px) {
            .main-title {
                font-size: 1.8rem;
            }
            .message.user, .message.bot {
                max-width: 100%;
            }
            .image-gallery img {
                max-width: 100%;
            }
        }
    </style>
    <script>
        const theme = localStorage.getItem('theme') || 'light';
        document.body.setAttribute('data-theme', theme);
        document.addEventListener('DOMContentLoaded', () => {
            const toggle = document.querySelector('input[type="checkbox"]');
            if (toggle) toggle.checked = theme === 'dark';
            document.querySelectorAll('button').forEach(btn => {
                if (btn.textContent.trim() === 'Clear History') {
                    btn.addEventListener('click', (e) => {
                        if (!confirm('Are you sure you want to clear all history? This action cannot be undone.')) {
                            e.preventDefault();
                        }
                    });
                }
            });
            const observeBotMessages = () => {
                const targetNode = document.querySelector('.main-content');
                const config = { childList: true, subtree: true };
                const callback = (mutationsList, observer) => {
                    for (let mutation of mutationsList) {
                        if (mutation.addedNodes.length) {
                            mutation.addedNodes.forEach(node => {
                                if (node.className && node.className.includes('message bot')) {
                                    const text = node.textContent.replace(/<[^>]+>/g, '').trim();
                                    if (text && !node.dataset.typed) {
                                        node.dataset.typed = 'true';
                                        node.textContent = '';
                                        let i = 0;
                                        const interval = setInterval(() => {
                                            if (i < text.length) {
                                                node.textContent += text.charAt(i);
                                                i++;
                                            } else {
                                                clearInterval(interval);
                                            }
                                        }, 50);
                                    }
                                }
                            });
                        }
                    }
                };
                const observer = new MutationObserver(callback);
                if (targetNode) observer.observe(targetNode, config);
                return observer;
            };
            const observer = observeBotMessages();
            const dropzone = document.querySelector('.stFileUploader');
            if (dropzone) {
                const dz = dropzone.closest('div').querySelector('div');
                dz.classList.add('upload-dropzone');
                dz.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    dz.classList.add('dragover');
                });
                dz.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    dz.classList.remove('dragover');
                });
                dz.addEventListener('drop', (e) => {
                    e.preventDefault();
                    dz.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    const dataTransfer = new DataTransfer();
                    for (let file of files) {
                        dataTransfer.items.add(file);
                    }
                    dropzone.querySelector('input[type="file"]').files = dataTransfer.files;
                    dropzone.dispatchEvent(new Event('change'));
                });
            }
            document.querySelectorAll('.image-gallery img').forEach(img => {
                img.addEventListener('click', () => {
                    const modal = document.createElement('div');
                    modal.style = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: flex; justify-content: center; align-items: center; z-index: 1000;';
                    const largeImg = document.createElement('img');
                    largeImg.src = img.src;
                    largeImg.style = 'max-width: 90%; max-height: 90%;';
                    modal.appendChild(largeImg);
                    modal.addEventListener('click', () => modal.remove());
                    document.body.appendChild(modal);
                });
            });
        });
    </script>
""", unsafe_allow_html=True)

# ========== Apply Theme ==========
if st.session_state["dark_mode"]:
    st.markdown('<body data-theme="dark"></body>', unsafe_allow_html=True)
else:
    st.markdown('<body data-theme="light"></body>', unsafe_allow_html=True)

# ========== Paths ==========
pdf_folder = "docs"
persist_dir = "chroma_db"
try:
    Path(pdf_folder).mkdir(exist_ok=True)
    Path(persist_dir).mkdir(exist_ok=True)
except PermissionError as e:
    st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Permission denied when creating directories: {e}</div>', unsafe_allow_html=True)
    st.stop()

# ========== Sidebar (Control Panel) ==========
with st.sidebar:
    st.markdown("<h2><i class='fas fa-cog'></i> Control Panel</h2>", unsafe_allow_html=True)
    
    # PDF Upload with Drag-and-Drop
    with st.expander("Upload Documents", expanded=True):
        st.markdown("<i class='fas fa-upload'></i> Upload PDF files", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, help="Upload PDFs to process and embed. Drag and drop is supported!", key="pdf_uploader")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = re.sub(r'[^\w\-\.]', '_', uploaded_file.name)
                file_path = Path(pdf_folder) / filename
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.markdown(f'<div class="custom-success"><i class="fas fa-check-circle"></i> Uploaded: {filename}</div>', unsafe_allow_html=True)
                    with st.spinner("Embedding uploaded file..."):
                        chunks = load_and_split_pdfs([uploaded_file], is_uploaded_files=True)
                        if chunks:
                            embed_and_store(chunks, persist_dir)
                            st.markdown(f'<div class="custom-success"><i class="fas fa-check-circle"></i> Embedded {len(chunks)} chunks from {filename}!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="custom-warning"><i class="fas fa-exclamation-triangle"></i> No content found in {filename}.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Failed to process {filename}: {e}</div>', unsafe_allow_html=True)
    
    # Website Scraping
    with st.expander("Scrape Website"):
        st.markdown("<i class='fas fa-globe'></i> Website Content", unsafe_allow_html=True)
        website_url = st.text_input("Website URL", placeholder="https://example.com", help="Enter a valid URL to scrape content.", key="website_url")
        if st.button("⬇️ Load Website", help="Scrape and save website content", key="load_website"):
            if not website_url.startswith(('http://', 'https://')):
                st.markdown('<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Invalid URL. Please include http:// or https://</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Scraping website..."):
                    try:
                        response = requests.get(website_url, timeout=10)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        text_content = soup.get_text(separator=' ', strip=True)
                        if len(text_content) > 10000:
                            st.markdown('<div class="custom-warning"><i class="fas fa-exclamation-triangle"></i> Content truncated to 10,000 characters.</div>', unsafe_allow_html=True)
                            text_content = text_content[:10000]
                        filename = f"website_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        Path(pdf_folder, filename).write_text(text_content, encoding="utf-8")
                        st.markdown(f'<div class="custom-success"><i class="fas fa-check-circle"></i> Website content saved as {filename}.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Error scraping website: {e}</div>', unsafe_allow_html=True)
    
    # Embed Content
    with st.expander("Embed Content"):
        st.markdown("<i class='fas fa-database'></i> Process Content", unsafe_allow_html=True)
        if st.button("⚙️ Process and Embed", help="Embed all uploaded content into the database", key="process_embed"):
            with st.spinner("Embedding content..."):
                try:
                    chunks = load_and_split_pdfs(pdf_folder, is_uploaded_files=False)
                    if not chunks:
                        st.markdown('<div class="custom-warning"><i class="fas fa-exclamation-triangle"></i> No content found to embed.</div>', unsafe_allow_html=True)
                    else:
                        progress_bar = st.progress(0)
                        batch_size = 100
                        for i in range(0, len(chunks), batch_size):
                            batch = chunks[i:i + batch_size]
                            embed_and_store(batch, persist_dir)
                            progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))
                        st.markdown(f'<div class="custom-success"><i class="fas fa-check-circle"></i> Embedded {len(chunks)} chunks!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Embedding failed: {e}</div>', unsafe_allow_html=True)
    
    # Appearance
    with st.expander("Appearance"):
        st.markdown("<i class='fas fa-paint-brush'></i> Theme Settings", unsafe_allow_html=True)
        dark_mode = st.toggle("Dark Mode", value=st.session_state["dark_mode"], help="Toggle between light and dark themes", key="theme_toggle")
        if dark_mode != st.session_state["dark_mode"]:
            st.session_state["dark_mode"] = dark_mode
            st.rerun()
    
    # Clear History
    st.markdown("<i class='fas fa-trash-alt'></i> Clear History", unsafe_allow_html=True)
    if st.button("Clear History", help="Reset chat and query cache", key="clear_history"):
        st.session_state.qa_history = []
        st.session_state.query_cache = OrderedDict()
        st.session_state.images = []
        st.markdown('<div class="custom-success"><i class="fas fa-check-circle"></i> History cleared.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by Groq, Hugging Face, ChromaDB, and xAI")

# ========== Main Content ==========
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown('<h1 class="main-title"><i class="fas fa-brain"></i> RAGvisor: Your AI-Powered Knowledge Hub</h1>', unsafe_allow_html=True)

# ... (previous imports and code up to Q&A Section remain unchanged)

# Q&A Section
with st.container():
    st.markdown("## <i class='fas fa-comments'></i> Ask a Question", unsafe_allow_html=True)
    query = st.text_input("Your Question", placeholder="Ask about your PDFs or websites...", help="Enter a question to query your documents", key="query_input")
    if st.button("✈️ Submit", help="Submit your question", key="submit_query"):
        query = bleach.clean(query.strip(), tags=[], strip=True)
        if len(query) < 3:
            st.markdown('<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Query must be at least 3 characters long.</div>', unsafe_allow_html=True)
        else:
            st.session_state.qa_history.append({"type": "user", "text": query})
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in st.session_state.query_cache:
                answer, documents, metadatas = st.session_state.query_cache[query_hash]
                st.session_state.query_cache.move_to_end(query_hash)
            else:
                try:
                    model = st.session_state.get("embedding_model", SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"))
                    st.session_state["embedding_model"] = model
                    client = PersistentClient(path=persist_dir)
                    collection = client.get_or_create_collection("rag_pdf")
                    query_embedding = model.encode(query, show_progress_bar=False).tolist()
                    results = collection.query(query_embeddings=[query_embedding], n_results=3)
                    documents = results.get("documents", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)
                    if not documents:
                        answer = "No relevant information found in the database for your query."
                        documents = []
                        metadatas = []
                    else:
                        context = "\n\n".join(documents)
                        with st.spinner("Generating answer..."):
                            answer = generate_answer(query, context)
                    st.session_state.query_cache[query_hash] = (answer, documents, metadatas)
                    if len(st.session_state.query_cache) > 100:
                        st.session_state.query_cache.popitem(last=False)
                except Exception as e:
                    st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Query failed: {e}</div>', unsafe_allow_html=True)
                    answer = "Sorry, I couldn't process your query due to an error."
                    documents = []
                    metadatas = []
            st.session_state.qa_history.append({"type": "bot", "text": answer})
            time.sleep(0.1)

# Chat History with Toggle
with st.container():
    if st.button("Toggle Chat History", key="toggle_chat"):
        st.session_state["chat_history_visible"] = not st.session_state["chat_history_visible"]
        st.rerun()
    if st.session_state["chat_history_visible"]:
        st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-history'></i> Conversation History", unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.qa_history):
            role = "user" if msg["type"] == "user" else "bot"
            icon = "<i class='fas fa-user'></i>" if role == "user" else "<i class='fas fa-robot'></i>"
            with st.container():
                st.markdown(f"<div class='message {role}'>{icon} {msg['text']}</div>", unsafe_allow_html=True)
                if st.button("📋 Copy", help="Copy message to clipboard", key=f"copy_{i}_{hashlib.md5(msg['text'].encode()).hexdigest()}"):
                    escaped_text = msg["text"].replace('"', '\\"').replace('\n', '\\n')
                    st.markdown(f'<script>navigator.clipboard.writeText("{escaped_text}");</script>', unsafe_allow_html=True)
                    st.markdown('<div class="custom-success"><i class="fas fa-check-circle"></i> Text copied to clipboard!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-history-container" style="display: none;"></div>', unsafe_allow_html=True)

# Retrieved Chunks
if "documents" in locals() and documents and metadatas:
    with st.expander("Retrieved Documents"):
        st.markdown("<i class='fas fa-book'></i> Retrieved Document Chunks", unsafe_allow_html=True)
        for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
            if metadata is None:
                metadata = {}
            source = metadata.get('source', 'Unknown Source')
            chunk_id = metadata.get('chunk_id', 'Unknown ID')
            st.markdown(f"<div class='chunk-card'><strong>Chunk {i} (Source: {source}, ID: {chunk_id}):</strong><br>{doc}</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="custom-warning"><i class="fas fa-exclamation-triangle"></i> No documents retrieved or metadata missing.</div>', unsafe_allow_html=True)

# ... (rest of the code remains unchanged, including Image Generation, Image Gallery, Footer, and JavaScript)
# Image Generation Section (Using DeepAI)
with st.container():
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.markdown("## <i class='fas fa-image'></i> Image Generation", unsafe_allow_html=True)
    img_prompt = st.text_input("Image Prompt", placeholder="Describe an image to generate...", help="Enter a prompt for image generation", key="img_prompt")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if st.button("✨ Use Last Answer", help="Generate image from the last bot response", key="use_last_answer"):
            last_answer = next((x['text'] for x in reversed(st.session_state.qa_history) if x['type'] == 'bot'), None)
            if last_answer:
                st.session_state["img_prompt"] = last_answer
                st.rerun()
            else:
                st.markdown('<div class="custom-warning"><i class="fas fa-exclamation-triangle"></i> No bot response available.</div>', unsafe_allow_html=True)
    with col_img2:
        if st.button("🎨 Generate Image", help="Generate image from prompt", key="generate_image"):
            img_prompt = bleach.clean(img_prompt.strip(), tags=[], strip=True)
            if len(img_prompt) < 5:
                st.markdown('<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Image prompt must be at least 5 characters long.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Generating image..."):
                    try:
                        deepai_api_key = os.getenv("DEEPAI_API_KEY")
                        headers = {"api-key": deepai_api_key}
                        api_url = "https://api.deepai.org/api/text2img"
                        payload = {"text": img_prompt}
                        response = requests.post(api_url, data=payload, headers=headers, timeout=30)
                        response.raise_for_status()
                        img_url = response.json().get("output_url")
                        if img_url:
                            img_response = requests.get(img_url, timeout=30)
                            img_response.raise_for_status()
                            image_bytes = BytesIO(img_response.content)
                            image = Image.open(image_bytes)
                            st.image(image, caption=f"🖼️ Generated Image: {img_prompt[:50]}...", use_container_width=True, output_format="PNG")
                            image_id = hashlib.md5(img_prompt.encode()).hexdigest()
                            st.session_state.images.append({"id": image_id, "prompt": img_prompt, "image": image_bytes.getvalue()})
                            st.download_button("⬇️ Download Image", data=image_bytes.getvalue(), file_name=f"generated_{image_id}.png", mime="image/png", key=f"download_image_{image_id}")
                        else:
                            st.markdown('<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Failed to retrieve image URL.</div>', unsafe_allow_html=True)
                    except requests.exceptions.HTTPError as e:
                        if response.status_code == 429:
                            st.markdown('<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Rate limit exceeded. Try again later or sign up for a free DeepAI account: <a href="https://deepai.org">DeepAI</a>.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Image generation failed: {e}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="custom-error"><i class="fas fa-exclamation-circle"></i> Image generation failed: {e}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Image Gallery
if st.session_state.images:
    with st.container():
        st.markdown("## <i class='fas fa-images'></i> Image Gallery", unsafe_allow_html=True)
        with st.expander("View Generated Images"):
            st.markdown("<i class='fas fa-images'></i> Generated Images", unsafe_allow_html=True)
            page = st.number_input("Page", min_value=1, max_value=(len(st.session_state.images) // 9) + 1, value=1, key="image_page")
            start_idx = (page - 1) * 9
            end_idx = start_idx + 9
            cols = st.columns(3)
            for i, img_data in enumerate(st.session_state.images[start_idx:end_idx]):
                with cols[i % 3]:
                    st.image(img_data["image"], caption=img_data["prompt"][:50] + "...", use_container_width=True, output_format="PNG")
                    st.download_button("⬇️ Download", data=img_data["image"], file_name=f"generated_{img_data['id']}.png", mime="image/png", key=f"download_gallery_{i}_{img_data['id']}")

# Footer
st.markdown("""
    <hr style="margin-top: 2rem; border-color: var(--primary-color);"/>
    <div style='text-align: center; font-size: 0.9rem; color: var(--text-color);'>
        Built with <i class='fas fa-heart' style='color: var(--accent-color);'></i> using Streamlit, Groq, Hugging Face, ChromaDB, and xAI — 2025
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# JavaScript for chat history toggle (simplified as button handles it in Python)
st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', () => container.style.display = {display});
    </script>
""".format(display='block' if st.session_state["chat_history_visible"] else 'none'), unsafe_allow_html=True)