# 🧠 RAGvisor

**RAGvisor** is a user-friendly, Streamlit-powered application that brings **Retrieval-Augmented Generation (RAG)** to your documents and web content. Ask natural language questions, and get AI-powered answers grounded in your PDF or scraped website content — complete with source citations and visual generation.


---

## 🚀 Features

-  **PDF & Website Ingestion**  
  Upload documents or scrape web pages. RAGvisor breaks them down into semantic chunks using Sentence Transformers.

-  **Semantic Search + Contextual QA**  
  Ask questions in plain English. Answers are generated via Groq’s Mixtral LLM with matching context retrieved from ChromaDB.

-  **AI Image Generator**  
  Visualize concepts with text-to-image prompts powered by DeepAI.

-  **RAG Architecture**  
  Combines document embeddings with fast inference using Groq's blazing LLM API.

-  **Modular Backend**  
  Built with clean, extensible Python modules for loaders, embedders, and LLM interaction.

---

##  Tech Stack

| Component       | Technology                       |
|----------------|----------------------------------|
| Framework       | Streamlit                        |
| LLM             | Mixtral-8x7B via Groq API        |
| Embedding       | SentenceTransformers (`MiniLM`) |
| Vector Store    | ChromaDB (PersistentClient)      |
| Image Generator | DeepAI API                       |
| Scraping        | BeautifulSoup + Requests         |

---

##  Demo Preview

![app-preview](Screenshot (93).png)
(Screenshot (95).png)

---

## 🔧 Installation

```bash
git clone https://github.com/TheharshVardhan01/RAGvisor.git
cd RAGvisor
pip install -r requirements.txt

Create a .env file in the root directory with:
GROQ_API_KEY=your_groq_key_here
DEEPAI_API_KEY=your_deepai_key_here



