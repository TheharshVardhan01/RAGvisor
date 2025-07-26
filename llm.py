import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ✅ Load API key from .env
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

DEFAULT_MODEL = "llama3-70b-8192"

def generate_answer(question, context, model=DEFAULT_MODEL):
    """
    Generate an answer using a Groq-hosted LLM with given context.
    Args:
        question (str): User query.
        context (str): Retrieved RAG context.
        model (str): LLM model name.
    Returns:
        str: Generated answer.
    """
    try:
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question precisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for answering document-related questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[LLM Error] {e}"


if __name__ == "__main__":
    context = "RAG helps reduce hallucination in LLMs by retrieving grounded, relevant knowledge chunks from a database."
    question = "How does RAG help LLMs overcome hallucinations?"
    print("Answer:\n", generate_answer(question, context))
