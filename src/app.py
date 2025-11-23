# import os
# import streamlit as st
# import tempfile
# import json
# from datetime import datetime

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_qdrant import QdrantVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_qdrant import Qdrant
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# from langchain_core.prompts import ChatPromptTemplate
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from dotenv import load_dotenv

# # --- Configuration ---
# load_dotenv()
# COLLECTION_NAME = "streamlit_pdf_rag"
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
# LOCAL_EMBEDDING_PATH = os.getenv("LOCAL_EMBEDDING_PATH", None)
# VECTOR_SIZE = 384

# # Available models for user selection
# AVAILABLE_LLM_MODELS = {
#     "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
#     "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
#     "Qwen3 32B": "qwen/qwen3-32b",
#     "OpenAI 20B": "openai/gpt-oss-20b",
# }

# AVAILABLE_EMBEDDING_MODELS = {
#     "BGE Small (Fast, 384d)": "BAAI/bge-small-en-v1.5",
#     "MiniLM (Very Fast, 384d)": "sentence-transformers/all-MiniLM-L6-v2"
# }

# # --- Utility Functions ---

# @st.cache_resource
# def get_groq_llm(model_name, temperature):
#     """Initializes and caches the Groq Language Model."""
#     groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
#     if not groq_api_key:
#         st.error("GROQ_API_KEY not found. Please set it in .env or st.secrets.")
#         return None
#     return ChatGroq(temperature=temperature, model_name=model_name, groq_api_key=groq_api_key)

# @st.cache_resource
# def get_qdrant_client_and_store(embedding_model_name):
#     """
#     Initializes and caches the Qdrant client and embeddings object.
#     Returns (client, embeddings, qdrant_url, qdrant_api_key)
#     """
#     qdrant_url = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL", None)
#     qdrant_api_key = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY", None)

#     if not qdrant_url or not qdrant_api_key:
#         st.error("QDRANT_URL or QDRANT_API_KEY not found. Please set them.")
#         return None, None, None, None

#     model_source = LOCAL_EMBEDDING_PATH if LOCAL_EMBEDDING_PATH else embedding_model_name

#     embeddings = HuggingFaceEmbeddings(
#         model_name=model_source,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)

#     return client, embeddings, qdrant_url, qdrant_api_key

# def check_or_create_collection(client):
#     """Checks if the collection exists, deletes it if it does, and recreates it."""
#     try:
#         collections = client.get_collections().collections
#         existing_names = [c.name for c in collections]

#         if COLLECTION_NAME in existing_names:
#             client.delete_collection(collection_name=COLLECTION_NAME)

#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=VECTOR_SIZE,
#                 distance=Distance.COSINE
#             )
#         )

#         st.toast(f"✅ Qdrant collection '{COLLECTION_NAME}' created.")
#         return True
    
#     except Exception as e:
#         st.error(f"Error checking/creating Qdrant collection: {e}")
#         return False


# def ingest_documents(uploaded_files, client, embeddings, qdrant_url, qdrant_api_key, chunk_size, chunk_overlap):
#     """Loads, chunks, embeds, and stores documents in Qdrant."""
#     if not uploaded_files:
#         st.warning("Please upload PDF files first.")
#         return

#     if not check_or_create_collection(client):
#         return

#     all_docs = []

#     # User-configurable chunking settings
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     for uploaded_file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name

#         try:
#             loader = PyPDFLoader(tmp_file_path)
#             docs = loader.load_and_split(text_splitter)

#             for doc in docs:
#                 doc.metadata["source"] = uploaded_file.name
#                 doc.metadata["preview"] = doc.page_content[:100]

#             all_docs.extend(docs)
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {e}")
#         finally:
#             try:
#                 os.unlink(tmp_file_path)
#             except Exception:
#                 pass

#     if not all_docs:
#         st.error("No content could be extracted from the uploaded files.")
#         return

#     try:
#         Qdrant.from_documents(
#             documents=all_docs,
#             embedding=embeddings,
#             url=qdrant_url,
#             api_key=qdrant_api_key,
#             collection_name=COLLECTION_NAME,
#             prefer_grpc=False
#         )
#         st.success(f"Successfully stored {len(all_docs)} chunks from {len(uploaded_files)} files in Qdrant!")
#         st.session_state.is_ingested = True
#     except Exception as e:
#         st.error(f"Error during embedding and Qdrant storage: {e}")


# # --- EVALUATION FUNCTIONS ---

# def evaluate_response(question, answer, context_docs, llm, embeddings):
#     """
#     Evaluates RAG response quality using LLM-as-judge approach.
#     Returns scores for: relevance, faithfulness, completeness, context_similarity
#     """
    
#     context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
#     # Calculate Context Similarity (Retrieval Accuracy) - IMPROVED
#     try:
#         question_embedding = embeddings.embed_query(question)
        
#         # Only use top 3 most relevant chunks for accuracy calculation
#         top_docs = context_docs[:3] if len(context_docs) > 3 else context_docs
#         context_embeddings = [embeddings.embed_query(doc.page_content) for doc in top_docs]
        
#         # Calculate cosine similarity for each retrieved doc
#         from numpy import dot
#         from numpy.linalg import norm
        
#         similarities = []
#         for ctx_emb in context_embeddings:
#             similarity = dot(question_embedding, ctx_emb) / (norm(question_embedding) * norm(ctx_emb))
#             # Normalize from [-1, 1] to [0, 100]
#             normalized_similarity = ((similarity + 1) / 2) * 100
#             similarities.append(normalized_similarity)
        
#         # Use maximum similarity (best match) instead of average
#         context_similarity = round(max(similarities), 1) if similarities else 0
#     except Exception as e:
#         context_similarity = 0
    
#     eval_prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are an expert evaluator assessing RAG system responses.
# Evaluate the answer based on:
# 1. RELEVANCE (0-10): Does the answer address the question?
# 2. FAITHFULNESS (0-10): Is the answer grounded in the provided context? No hallucinations?
# 3. COMPLETENESS (0-10): Is the answer comprehensive and detailed?

# Respond ONLY with valid JSON format:
# {{
#     "relevance": <score>,
#     "faithfulness": <score>,
#     "completeness": <score>,
#     "explanation": "<brief explanation>"
# }}"""),
#         ("human", """Question: {question}

# Context: {context}

# Answer: {answer}

# Evaluate this response:""")
#     ])
    
#     try:
#         chain = eval_prompt | llm
#         result = chain.invoke({
#             "question": question,
#             "context": context_text,
#             "answer": answer
#         })
        
#         eval_result = json.loads(result.content)
#         eval_result["context_similarity"] = context_similarity
#         return eval_result
    
#     except Exception as e:
#         return {
#             "relevance": 0,
#             "faithfulness": 0,
#             "completeness": 0,
#             "context_similarity": context_similarity,
#             "explanation": f"Evaluation error: {str(e)}"
#         }


# def save_evaluation(question, answer, sources, eval_scores):
#     """Saves evaluation results to a JSON file for tracking."""
    
#     eval_data = {
#         "timestamp": datetime.now().isoformat(),
#         "question": question,
#         "answer": answer,
#         "sources": sources,
#         "scores": eval_scores
#     }
    
#     eval_file = "rag_evaluations.json"
    
#     if os.path.exists(eval_file):
#         with open(eval_file, 'r') as f:
#             evaluations = json.load(f)
#     else:
#         evaluations = []
    
#     evaluations.append(eval_data)
    
#     with open(eval_file, 'w') as f:
#         json.dump(evaluations, f, indent=2)


# # --- MAIN APP ---

# def main():
#     """Main function for the Streamlit RAG App."""
#     st.set_page_config(page_title="PDF RAG Chatbot with Advanced Features", layout="wide")
#     st.title("QueryMaster: PDF RAG Chatbot with Advanced Features")
#     st.caption("Powered by Configurable Embeddings, Qdrant, and Groq LLM")

#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "is_ingested" not in st.session_state:
#         st.session_state.is_ingested = False
#     if "enable_evaluation" not in st.session_state:
#         st.session_state.enable_evaluation = False

#     # --- Sidebar ---
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # Model Selection
#         st.subheader("1. Model Selection")
#         selected_llm = st.selectbox(
#             "LLM Model",
#             options=list(AVAILABLE_LLM_MODELS.keys()),
#             help="Choose your language model"
#         )
#         llm_model_name = AVAILABLE_LLM_MODELS[selected_llm]
        
#         selected_embedding = st.selectbox(
#             "Embedding Model",
#             options=list(AVAILABLE_EMBEDDING_MODELS.keys()),
#             help="Choose your embedding model (requires re-ingestion)"
#         )
#         embedding_model_name = AVAILABLE_EMBEDDING_MODELS[selected_embedding]
        
#         # Hyperparameters
#         st.subheader("2. Hyperparameters")
        
#         temperature = st.slider(
#             "Temperature",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.0,
#             step=0.1,
#             help="0 = deterministic, 1 = creative"
#         )
        
#         chunk_size = st.slider(
#             "Chunk Size",
#             min_value=200,
#             max_value=2000,
#             value=800,
#             step=100,
#             help="Size of text chunks for embedding"
#         )
        
#         chunk_overlap = st.slider(
#             "Chunk Overlap",
#             min_value=0,
#             max_value=500,
#             value=200,
#             step=50,
#             help="Overlap between consecutive chunks"
#         )
        
#         k_docs = st.slider(
#             "Documents to Retrieve (k)",
#             min_value=1,
#             max_value=15,
#             value=7,
#             step=1,
#             help="Number of documents to retrieve"
#         )
        
#         fetch_k = st.slider(
#             "Candidate Pool (fetch_k)",
#             min_value=5,
#             max_value=50,
#             value=20,
#             step=5,
#             help="Number of candidates for MMR"
#         )
        
#         search_type = st.selectbox(
#             "Search Strategy",
#             options=["mmr", "similarity"],
#             help="MMR = diverse results, Similarity = most similar"
#         )

#         st.info("💡 API Keys should be set as environment variables or in `st.secrets`.")

#         # Initialize clients with selected models
#         client, embeddings, qdrant_url, qdrant_api_key = get_qdrant_client_and_store(embedding_model_name)
#         llm = get_groq_llm(llm_model_name, temperature)

#         if not client or not embeddings or not llm:
#             st.error("API Keys / Qdrant connection missing or LLM not configured.")
#         else:
#             st.header("📂 Document Ingestion")
#             uploaded_files = st.file_uploader(
#                 "Upload your PDF files",
#                 type=["pdf"],
#                 accept_multiple_files=True
#             )

#             if st.button("🚀 Embed & Store Documents", disabled=not uploaded_files):
#                 with st.spinner("Processing documents..."):
#                     ingest_documents(uploaded_files, client, embeddings, qdrant_url, qdrant_api_key, chunk_size, chunk_overlap)

#             if st.session_state.is_ingested:
#                 st.success("✅ Data is ready for chat!")
#             else:
#                 st.warning("⚠️ Upload PDFs and click 'Embed & Store Documents' to begin.")

#             # Evaluation Settings
#             st.header("📊 Evaluation")
#             st.session_state.enable_evaluation = st.checkbox(
#                 "Enable Response Evaluation",
#                 value=st.session_state.enable_evaluation,
#                 help="Evaluate each response (uses extra LLM calls)"
#             )
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 if st.button("📈 View Report"):
#                     if os.path.exists("rag_evaluations.json"):
#                         with open("rag_evaluations.json", 'r') as f:
#                             evals = json.load(f)
                        
#                         if evals:
#                             # All time metrics
#                             avg_relevance = sum(e["scores"]["relevance"] for e in evals) / len(evals)
#                             avg_faithfulness = sum(e["scores"]["faithfulness"] for e in evals) / len(evals)
#                             avg_completeness = sum(e["scores"]["completeness"] for e in evals) / len(evals)
#                             avg_similarity = sum(e["scores"].get("context_similarity", 0) for e in evals) / len(evals)
                            
#                             st.success(f"""
# **All Time Metrics ({len(evals)} responses)**
# - Avg Relevance: {avg_relevance:.1f}/10
# - Avg Faithfulness: {avg_faithfulness:.1f}/10
# - Avg Completeness: {avg_completeness:.1f}/10
# - Avg Retrieval Accuracy: {avg_similarity:.1f}%
#                             """)
                            
#                             # Recent metrics
#                             recent = evals[-3:] if len(evals) > 3 else evals
#                             recent_similarity = sum(e["scores"].get("context_similarity", 0) for e in recent) / len(recent)
#                             st.info(f"""
# **Recent Performance (last {len(recent)})**
# - Retrieval Accuracy: {recent_similarity:.1f}%
#                             """)
#                         else:
#                             st.info("No evaluations yet.")
#                     else:
#                         st.info("No evaluations file found.")
            
#             with col2:
#                 if st.button("🗑️ Clear History"):
#                     if os.path.exists("rag_evaluations.json"):
#                         os.remove("rag_evaluations.json")
#                         st.success("History cleared!")
#                         st.rerun()
#                     else:
#                         st.info("No history to clear.")

#     # --- Chat Interface ---
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.is_ingested):
#         if not client or not embeddings or not llm:
#             st.error("Cannot chat. Please check your API keys and Qdrant connection.")
#             return

#         if not st.session_state.is_ingested:
#             st.error("Please ingest documents before chatting.")
#             return

#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Setup Retriever with user-configured parameters
#         qdrant_vectorstore = QdrantVectorStore(
#             client=client,
#             collection_name=COLLECTION_NAME,
#             embedding=embeddings
#         )

#         # Configure retriever based on search type
#         if search_type == "mmr":
#             retriever = qdrant_vectorstore.as_retriever(
#                 search_type="mmr",
#                 search_kwargs={"k": k_docs, "fetch_k": fetch_k}
#             )
#         else:
#             retriever = qdrant_vectorstore.as_retriever(
#                 search_type=search_type,
#                 search_kwargs={"k": k_docs}
#             )

#         system_prompt = (
#             "You are an expert RAG assistant helping answer questions about the ADSAI curriculum. "
#             "Answer the user's question using ONLY the provided context. Be SPECIFIC and DETAILED. "
#             "If the context mentions specific courses, modules, tracks, requirements, or details, include them in your answer. "
#             "If the context does not contain the answer, state clearly that the information is not available. "
#             "Do not make up information or sources.\n\n"
#             "Context: {context}"
#         )

#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "{input}")
#         ])

#         try:
#             document_chain = create_stuff_documents_chain(llm, prompt_template)
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     try:
#                         response = retrieval_chain.invoke({"input": prompt})

#                         full_response = response.get("answer") or response.get("output_text") or str(response)
#                         source_docs = response.get("context") or []

#                         sources = "\n".join(
#                             [f'- **{doc.metadata.get("source", "Unknown Source")}** (Page {doc.metadata.get("page", "N/A")})' 
#                              for doc in source_docs]
#                         )

#                         # final_output = f"{full_response}\n\n---\n**Sources Used:**\n{sources}"
#                         final_output = f"{full_response}"
#                         st.markdown(final_output)
#                         st.session_state.messages.append({"role": "assistant", "content": final_output})

#                         # EVALUATION
#                         if st.session_state.enable_evaluation:
#                             with st.spinner("Evaluating response quality..."):
#                                 eval_scores = evaluate_response(
#                                     question=prompt,
#                                     answer=full_response,
#                                     context_docs=source_docs,
#                                     llm=llm,
#                                     embeddings=embeddings
#                                 )
                                
#                                 st.markdown("### 📊 Response Quality Scores")
#                                 col1, col2, col3, col4 = st.columns(4)
                                
#                                 with col1:
#                                     relevance_color = "🟢" if eval_scores["relevance"] >= 7 else "🟡" if eval_scores["relevance"] >= 5 else "🔴"
#                                     st.metric("Relevance", f"{eval_scores['relevance']}/10 {relevance_color}")
                                
#                                 with col2:
#                                     faithfulness_color = "🟢" if eval_scores["faithfulness"] >= 7 else "🟡" if eval_scores["faithfulness"] >= 5 else "🔴"
#                                     st.metric("Faithfulness", f"{eval_scores['faithfulness']}/10 {faithfulness_color}")
                                
#                                 with col3:
#                                     completeness_color = "🟢" if eval_scores["completeness"] >= 7 else "🟡" if eval_scores["completeness"] >= 5 else "🔴"
#                                     st.metric("Completeness", f"{eval_scores['completeness']}/10 {completeness_color}")
                                
#                                 with col4:
#                                     similarity = eval_scores.get("context_similarity", 0)
#                                     similarity_color = "🟢" if similarity >= 70 else "🟡" if similarity >= 50 else "🔴"
#                                     st.metric("Retrieval Accuracy", f"{similarity:.1f}% {similarity_color}")
                                
#                                 with st.expander("📝 Evaluation Details"):
#                                     st.write(eval_scores.get("explanation", "No explanation provided"))
#                                     st.write(f"\n**Retrieval Accuracy**: Measures semantic similarity between question and retrieved context.")
#                                     st.write(f"**Current Settings**: Model={selected_llm}, Temp={temperature}, k={k_docs}, Strategy={search_type}")
                                
#                                 # Save evaluation
#                                 source_list = [f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" 
#                                               for doc in source_docs]
#                                 save_evaluation(prompt, full_response, source_list, eval_scores)

#                     except Exception as e:
#                         st.error(f"An error occurred during chat: {e}")
#                         st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
        
#         except Exception as e:
#             st.warning("Falling back to manual retrieval.")
#             try:
#                 docs = retriever.get_relevant_documents(prompt)
#                 context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
                
#                 composed_prompt = (
#                     "You are an expert RAG assistant. Answer based only on the provided context.\n"
#                     f"Context:\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
#                 )

#                 try:
#                     llm_response = llm.generate([{"role": "user", "content": composed_prompt}])
#                     text_out = None
#                     if hasattr(llm_response, "generations"):
#                         try:
#                             text_out = llm_response.generations[0][0].text
#                         except Exception:
#                             text_out = str(llm_response)
#                     else:
#                         text_out = str(llm_response)
#                 except Exception:
#                     try:
#                         text_out = llm(composed_prompt)
#                         text_out = text_out if isinstance(text_out, str) else str(text_out)
#                     except Exception as e2:
#                         raise RuntimeError(f"LLM invocation failed: {e2}")

#                 sources = "\n".join(
#                     [f'- **{doc.metadata.get("source", "Unknown Source")}** (Page {doc.metadata.get("page", "N/A")})' 
#                      for doc in docs]
#                 )
#                 final_output = f"{text_out}\n\n---\n**Sources Used:**\n{sources}"
#                 st.markdown(final_output)
#                 st.session_state.messages.append({"role": "assistant", "content": final_output})
#             except Exception as e2:
#                 st.error(f"Fallback flow failed: {e2}")
#                 st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e2}"})


# if __name__ == "__main__":
#     main()
import os
import streamlit as st
import tempfile
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

# --- LOGGING SETUP ---
def setup_logging():
    """Configure comprehensive logging system"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create logger
    logger = logging.getLogger('RAG_Chatbot')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File Handler - Main log (rotates at 10MB, keeps 5 backups)
    file_handler = RotatingFileHandler(
        'logs/rag_chatbot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # File Handler - Error log (only errors and critical)
    error_handler = RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # File Handler - Query log (user interactions)
    query_handler = RotatingFileHandler(
        'logs/queries.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    query_handler.setLevel(logging.INFO)
    query_formatter = logging.Formatter(
        '%(asctime)s - QUERY - %(message)s'
    )
    query_handler.setFormatter(query_formatter)
    
    # Console Handler (optional - for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger, query_handler

# Initialize logger
logger, query_handler = setup_logging()
query_logger = logging.getLogger('QueryLogger')
query_logger.addHandler(query_handler)
query_logger.setLevel(logging.INFO)

# --- Configuration ---
load_dotenv()
COLLECTION_NAME = "streamlit_pdf_rag"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
LOCAL_EMBEDDING_PATH = os.getenv("LOCAL_EMBEDDING_PATH", None)
VECTOR_SIZE = 384

logger.info("="*50)
logger.info("RAG Chatbot Application Started")
logger.info(f"Collection Name: {COLLECTION_NAME}")
logger.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

# Available models for user selection
AVAILABLE_LLM_MODELS = {
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
    "Qwen3 32B": "qwen/qwen3-32b",
    "OpenAI 20B": "openai/gpt-oss-20b",
}

AVAILABLE_EMBEDDING_MODELS = {
    "BGE Small (Fast, 384d)": "BAAI/bge-small-en-v1.5",
    "MiniLM (Very Fast, 384d)": "sentence-transformers/all-MiniLM-L6-v2"
}

# --- Utility Functions ---

@st.cache_resource
def get_groq_llm(model_name, temperature):
    """Initializes and caches the Groq Language Model."""
    logger.info(f"Initializing Groq LLM: model={model_name}, temperature={temperature}")
    
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found")
        st.error("GROQ_API_KEY not found. Please set it in .env or st.secrets.")
        return None
    
    try:
        llm = ChatGroq(temperature=temperature, model_name=model_name, groq_api_key=groq_api_key)
        logger.info(f"Successfully initialized Groq LLM: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}", exc_info=True)
        return None

@st.cache_resource
def get_qdrant_client_and_store(embedding_model_name):
    """Initializes and caches the Qdrant client and embeddings object."""
    logger.info(f"Initializing Qdrant client and embeddings: {embedding_model_name}")
    
    qdrant_url = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL", None)
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY", None)

    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL or QDRANT_API_KEY not found")
        st.error("QDRANT_URL or QDRANT_API_KEY not found. Please set them.")
        return None, None, None, None

    try:
        model_source = LOCAL_EMBEDDING_PATH if LOCAL_EMBEDDING_PATH else embedding_model_name

        embeddings = HuggingFaceEmbeddings(
            model_name=model_source,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Embeddings model loaded: {model_source}")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
        logger.info(f"Qdrant client connected: {qdrant_url}")
        
        return client, embeddings, qdrant_url, qdrant_api_key
    
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant/Embeddings: {e}", exc_info=True)
        return None, None, None, None

def check_or_create_collection(client):
    """Checks if the collection exists, deletes it if it does, and recreates it."""
    logger.info(f"Checking/creating collection: {COLLECTION_NAME}")
    
    try:
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]
        logger.debug(f"Existing collections: {existing_names}")

        if COLLECTION_NAME in existing_names:
            logger.warning(f"Collection {COLLECTION_NAME} exists, deleting...")
            client.delete_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        logger.info(f"Collection {COLLECTION_NAME} created successfully")
        st.toast(f"✅ Qdrant collection '{COLLECTION_NAME}' created.")
        return True
    
    except Exception as e:
        logger.error(f"Error creating collection: {e}", exc_info=True)
        st.error(f"Error checking/creating Qdrant collection: {e}")
        return False


def ingest_documents(uploaded_files, client, embeddings, qdrant_url, qdrant_api_key, chunk_size, chunk_overlap):
    """Loads, chunks, embeds, and stores documents in Qdrant."""
    logger.info(f"Starting document ingestion: {len(uploaded_files)} files")
    logger.info(f"Chunking params: size={chunk_size}, overlap={chunk_overlap}")
    
    if not uploaded_files:
        logger.warning("No files uploaded")
        st.warning("Please upload PDF files first.")
        return

    if not check_or_create_collection(client):
        return

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for uploaded_file in uploaded_files:
        logger.info(f"Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load_and_split(text_splitter)
            
            logger.info(f"Extracted {len(docs)} chunks from {uploaded_file.name}")

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["preview"] = doc.page_content[:100]

            all_docs.extend(docs)
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

    if not all_docs:
        logger.error("No content extracted from files")
        st.error("No content could be extracted from the uploaded files.")
        return

    try:
        logger.info(f"Embedding {len(all_docs)} chunks...")
        Qdrant.from_documents(
            documents=all_docs,
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
            prefer_grpc=False
        )
        
        logger.info(f"Successfully stored {len(all_docs)} chunks in Qdrant")
        st.success(f"Successfully stored {len(all_docs)} chunks from {len(uploaded_files)} files in Qdrant!")
        st.session_state.is_ingested = True
        
    except Exception as e:
        logger.error(f"Error during embedding/storage: {e}", exc_info=True)
        st.error(f"Error during embedding and Qdrant storage: {e}")


# --- EVALUATION FUNCTIONS ---

def evaluate_response(question, answer, context_docs, llm, embeddings):
    """Evaluates RAG response quality using LLM-as-judge approach."""
    logger.info(f"Evaluating response for question: {question[:50]}...")
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Calculate Context Similarity
    try:
        question_embedding = embeddings.embed_query(question)
        top_docs = context_docs[:3] if len(context_docs) > 3 else context_docs
        context_embeddings = [embeddings.embed_query(doc.page_content) for doc in top_docs]
        
        from numpy import dot
        from numpy.linalg import norm
        
        similarities = []
        for ctx_emb in context_embeddings:
            similarity = dot(question_embedding, ctx_emb) / (norm(question_embedding) * norm(ctx_emb))
            normalized_similarity = ((similarity + 1) / 2) * 100
            similarities.append(normalized_similarity)
        
        context_similarity = round(max(similarities), 1) if similarities else 0
        logger.debug(f"Context similarity calculated: {context_similarity}%")
        
    except Exception as e:
        logger.error(f"Error calculating context similarity: {e}", exc_info=True)
        context_similarity = 0
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator assessing RAG system responses.
Evaluate the answer based on:
1. RELEVANCE (0-10): Does the answer address the question?
2. FAITHFULNESS (0-10): Is the answer grounded in the provided context? No hallucinations?
3. COMPLETENESS (0-10): Is the answer comprehensive and detailed?

Respond ONLY with valid JSON format:
{{
    "relevance": <score>,
    "faithfulness": <score>,
    "completeness": <score>,
    "explanation": "<brief explanation>"
}}"""),
        ("human", """Question: {question}

Context: {context}

Answer: {answer}

Evaluate this response:""")
    ])
    
    try:
        chain = eval_prompt | llm
        result = chain.invoke({
            "question": question,
            "context": context_text,
            "answer": answer
        })
        
        eval_result = json.loads(result.content)
        eval_result["context_similarity"] = context_similarity
        
        logger.info(f"Evaluation scores - Relevance: {eval_result['relevance']}, "
                   f"Faithfulness: {eval_result['faithfulness']}, "
                   f"Completeness: {eval_result['completeness']}, "
                   f"Retrieval: {context_similarity}%")
        
        return eval_result
    
    except Exception as e:
        logger.error(f"Error during LLM evaluation: {e}", exc_info=True)
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "context_similarity": context_similarity,
            "explanation": f"Evaluation error: {str(e)}"
        }


def save_evaluation(question, answer, sources, eval_scores):
    """Saves evaluation results to a JSON file for tracking."""
    logger.debug("Saving evaluation to file")
    
    eval_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "scores": eval_scores
    }
    
    eval_file = "rag_evaluations.json"
    
    try:
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                evaluations = json.load(f)
        else:
            evaluations = []
        
        evaluations.append(eval_data)
        
        with open(eval_file, 'w') as f:
            json.dump(evaluations, f, indent=2)
        
        logger.info(f"Evaluation saved. Total evaluations: {len(evaluations)}")
    
    except Exception as e:
        logger.error(f"Error saving evaluation: {e}", exc_info=True)


# --- MAIN APP ---

def main():
    """Main function for the Streamlit RAG App."""
    st.set_page_config(page_title="QueryMaster - Advanced RAG", layout="wide")
    st.title("🤖 QueryMaster: Advanced RAG Chatbot")
    st.caption("Powered by Configurable Embeddings, Qdrant, and Groq LLM")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("New chat session started")
    if "is_ingested" not in st.session_state:
        st.session_state.is_ingested = False
    if "enable_evaluation" not in st.session_state:
        st.session_state.enable_evaluation = False

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("1. Model Selection")
        selected_llm = st.selectbox(
            "LLM Model",
            options=list(AVAILABLE_LLM_MODELS.keys()),
            help="Choose your language model"
        )
        llm_model_name = AVAILABLE_LLM_MODELS[selected_llm]
        
        selected_embedding = st.selectbox(
            "Embedding Model",
            options=list(AVAILABLE_EMBEDDING_MODELS.keys()),
            help="Choose your embedding model"
        )
        embedding_model_name = AVAILABLE_EMBEDDING_MODELS[selected_embedding]
        
        # Hyperparameters
        st.subheader("2. Hyperparameters")
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        chunk_size = st.slider("Chunk Size", 200, 2000, 800, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        k_docs = st.slider("Documents to Retrieve (k)", 1, 15, 7, 1)
        fetch_k = st.slider("Candidate Pool (fetch_k)", 5, 50, 20, 5)
        search_type = st.selectbox("Search Strategy", ["mmr", "similarity"])

        st.info("💡 API Keys in environment variables or st.secrets")

        # Initialize clients
        client, embeddings, qdrant_url, qdrant_api_key = get_qdrant_client_and_store(embedding_model_name)
        llm = get_groq_llm(llm_model_name, temperature)

        if not client or not embeddings or not llm:
            st.error("Configuration error - check logs")
        else:
            st.header("📂 Document Ingestion")
            uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

            if st.button("🚀 Embed & Store", disabled=not uploaded_files):
                with st.spinner("Processing..."):
                    ingest_documents(uploaded_files, client, embeddings, qdrant_url, qdrant_api_key, chunk_size, chunk_overlap)

            if st.session_state.is_ingested:
                st.success("✅ Ready for chat!")
            else:
                st.warning("⚠️ Upload & embed documents first")

            # Evaluation
            st.header("📊 Evaluation")
            st.session_state.enable_evaluation = st.checkbox("Enable Evaluation", value=st.session_state.enable_evaluation)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Report"):
                    if os.path.exists("rag_evaluations.json"):
                        with open("rag_evaluations.json", 'r') as f:
                            evals = json.load(f)
                        
                        if evals:
                            avg_rel = sum(e["scores"]["relevance"] for e in evals) / len(evals)
                            avg_faith = sum(e["scores"]["faithfulness"] for e in evals) / len(evals)
                            avg_comp = sum(e["scores"]["completeness"] for e in evals) / len(evals)
                            avg_sim = sum(e["scores"].get("context_similarity", 0) for e in evals) / len(evals)
                            
                            st.success(f"""**Metrics ({len(evals)} queries)**
- Relevance: {avg_rel:.1f}/10
- Faithfulness: {avg_faith:.1f}/10
- Completeness: {avg_comp:.1f}/10
- Retrieval: {avg_sim:.1f}%""")
                    else:
                        st.info("No evaluations yet")
            
            with col2:
                if st.button("🗑️ Clear"):
                    if os.path.exists("rag_evaluations.json"):
                        os.remove("rag_evaluations.json")
                        logger.info("Evaluation history cleared")
                        st.success("Cleared!")
                        st.rerun()
            
            with col3:
                if st.button("📋 Logs"):
                    if os.path.exists("logs/rag_chatbot.log"):
                        with open("logs/rag_chatbot.log", 'r') as f:
                            logs = f.readlines()[-50:]  # Last 50 lines
                        st.text_area("Recent Logs", "".join(logs), height=300)
                    else:
                        st.info("No logs found")

    # --- Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question...", disabled=not st.session_state.is_ingested):
        logger.info(f"User query received: {prompt}")
        query_logger.info(f"Q: {prompt}")
        
        if not client or not embeddings or not llm:
            st.error("Configuration error")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Setup Retriever
        qdrant_vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )

        if search_type == "mmr":
            retriever = qdrant_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_docs, "fetch_k": fetch_k}
            )
        else:
            retriever = qdrant_vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k_docs}
            )

        system_prompt = (
            "You are an expert RAG assistant. Answer using ONLY the provided context. "
            "Be SPECIFIC and DETAILED. If information is not in context, say so. "
            "Do not make up information.\n\nContext: {context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        try:
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        start_time = datetime.now()
                        response = retrieval_chain.invoke({"input": prompt})
                        end_time = datetime.now()
                        latency = (end_time - start_time).total_seconds()
                        
                        logger.info(f"Response generated in {latency:.2f}s")

                        full_response = response.get("answer") or response.get("output_text") or str(response)
                        source_docs = response.get("context") or []
                        
                        query_logger.info(f"A: {full_response[:100]}...")
                        logger.info(f"Retrieved {len(source_docs)} source documents")

                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                        # EVALUATION
                        if st.session_state.enable_evaluation:
                            with st.spinner("Evaluating..."):
                                eval_scores = evaluate_response(prompt, full_response, source_docs, llm, embeddings)
                                
                                st.markdown("### 📊 Quality Scores")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    color = "🟢" if eval_scores["relevance"] >= 7 else "🟡" if eval_scores["relevance"] >= 5 else "🔴"
                                    st.metric("Relevance", f"{eval_scores['relevance']}/10 {color}")
                                
                                with col2:
                                    color = "🟢" if eval_scores["faithfulness"] >= 7 else "🟡" if eval_scores["faithfulness"] >= 5 else "🔴"
                                    st.metric("Faithfulness", f"{eval_scores['faithfulness']}/10 {color}")
                                
                                with col3:
                                    color = "🟢" if eval_scores["completeness"] >= 7 else "🟡" if eval_scores["completeness"] >= 5 else "🔴"
                                    st.metric("Completeness", f"{eval_scores['completeness']}/10 {color}")
                                
                                with col4:
                                    sim = eval_scores.get("context_similarity", 0)
                                    color = "🟢" if sim >= 70 else "🟡" if sim >= 50 else "🔴"
                                    st.metric("Retrieval", f"{sim:.1f}% {color}")
                                
                                with st.expander("📝 Details"):
                                    st.write(eval_scores.get("explanation", "N/A"))
                                    st.write(f"**Latency**: {latency:.2f}s")
                                    st.write(f"**Settings**: {selected_llm}, temp={temperature}, k={k_docs}")
                                
                                source_list = [f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" 
                                              for doc in source_docs]
                                save_evaluation(prompt, full_response, source_list, eval_scores)

                    except Exception as e:
                        logger.error(f"Error during chat: {e}", exc_info=True)
                        st.error(f"Error: {e}")
        
        except Exception as e:
            logger.error(f"Chain error: {e}", exc_info=True)
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()