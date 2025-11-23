# QueryMaster — Advanced PDF RAG Chatbot

QueryMaster is a configurable Retrieval Augmented Generation (RAG) system built with Streamlit, Groq LLMs, Qdrant Vector DB, and HuggingFace Embeddings.
Upload PDFs, generate embeddings, and interact with your documents using an optimized retrieval pipeline.

https://github.com/user-attachments/assets/7ad29c9b-7c8b-4e52-bc19-30a6fc73dc1f

## Features

- PDF → Embeddings → Qdrant ingestion pipeline
- Configurable chunk size, overlap, k, fetch_k, MMR/similarity
- Supports Llama 3.1/3.3, Qwen3, GPT-OSS via Groq
- Built-in LLM-as-judge evaluation
- Logging system (errors, queries, runtime)
- Clean Streamlit UI with model + embedding selection

## RAG Optimization (Real Results)

You observed major improvements when tuning retrieval:
| Setting | Chunk Size | k | Retrieval Accuracy |
| ------- | ---------- | - | ------------------ |
| Before  | 1500       | 5 | 26%                |
| After   | 800        | 3 | 78%                |

This improvement happened because:

✔ Smaller chunk size (800)
- Creates more fine-grained embeddings
- Allows the retriever to match precise content
- Reduces "overly large" chunks that dilute semantic meaning

✔ Lower k (3)
- Reduces retrieval noise
- Ensures the LLM receives only the most relevant chunks
- Prevents irrelevant context from lowering evaluation scores

## Installation
1️⃣ Clone the repository
```git clone https://github.com/yourusername/querymaster-rag.git```
```cd querymaster-rag```

2️⃣ Install dependencies
```pip install -r requirements.txt```

3️⃣ Set environment variables
Create a .env file:
```
GROQ_API_KEY=your_key
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_key
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
```

4️⃣ Run the app
```
streamlit run app.py
```

## How to Use
1. Upload PDF files
The ingestion engine will:
- Split PDFs
- Embed text
- Store them in Qdrant
- Mark the dataset as “Ready for chat”

3. Configure parameters
You can adjust:
- LLM model
- Embedding model
- Chunk size
- Chunk overlap
- Retrieval k
- MMR fetch_k
- Search strategy (MMR / Similarity)

5. Ask questions
Ask domain-specific questions from the PDFs.

6. Evaluate responses
Enable "Response Evaluation" to see real-time scores.

## Project Structure

```
.
├── app.py                     # Main Streamlit app
├── rag_evaluations.json       # Auto-generated evaluation logs
├── logs/
│   ├── rag_chatbot.log
│   ├── errors.log
│   └── queries.log
├── requirements.txt
└── README.md
```

## Example Query Flow
- User asks: "What programs are offered in the ADSAI curriculum?"
- System retrieves relevant chunks from Qdrant
- Groq LLM generates an answer
- Evaluation module scores the response
- Metrics are displayed in UI

## RAG Optimization Tips (Based on Your Results)

- Chunk Size 800–1000 works best for academic PDFs
- k = 2–4 reduces noise
- Use MMR when PDFs have repetitive sections
- Keep temperature at 0.0 for factual content
- BGE-Small is a strong embedding model for 384-dim vectors

