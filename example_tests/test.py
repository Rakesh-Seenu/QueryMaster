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

# # --- Utility Functions ---

# @st.cache_resource
# def get_groq_llm():
#     """Initializes and caches the Groq Language Model."""
#     groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
#     if not groq_api_key:
#         st.error("GROQ_API_KEY not found. Please set it in .env or st.secrets.")
#         return None
#     return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# @st.cache_resource
# def get_qdrant_client_and_store():
#     """
#     Initializes and caches the Qdrant client and embeddings object.
#     Returns (client, embeddings, qdrant_url, qdrant_api_key)
#     """
#     qdrant_url = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL", None)
#     qdrant_api_key = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY", None)

#     if not qdrant_url or not qdrant_api_key:
#         st.error("QDRANT_URL or QDRANT_API_KEY not found. Please set them.")
#         return None, None, None, None

#     model_source = LOCAL_EMBEDDING_PATH if LOCAL_EMBEDDING_PATH else EMBEDDING_MODEL_NAME

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


# def ingest_documents(uploaded_files, client, embeddings, qdrant_url, qdrant_api_key):
#     """Loads, chunks, embeds, and stores documents in Qdrant."""
#     if not uploaded_files:
#         st.warning("Please upload PDF files first.")
#         return

#     if not check_or_create_collection(client):
#         return

#     all_docs = []

#     # Improved chunking settings for better retrieval
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,      # Smaller chunks = more focused content
#         chunk_overlap=200,   # Good overlap to preserve context
#         separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting
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
#     st.set_page_config(page_title="PDF RAG Chatbot (Local Embeddings + Groq)", layout="wide")
#     st.title("📄 PDF RAG Chatbot")
#     st.caption("Powered by Local HuggingFace Embeddings, Qdrant, and Groq LLM")

#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "is_ingested" not in st.session_state:
#         st.session_state.is_ingested = False
#     if "enable_evaluation" not in st.session_state:
#         st.session_state.enable_evaluation = False

#     # --- Sidebar ---
#     with st.sidebar:
#         st.header("1. Configuration")
#         st.info("Keys should be set as environment variables or in `st.secrets`.")

#         client, embeddings, qdrant_url, qdrant_api_key = get_qdrant_client_and_store()
#         llm = get_groq_llm()

#         if not client or not embeddings or not llm:
#             st.error("API Keys / Qdrant connection missing or LLM not configured. Check your configuration.")
#         else:
#             st.header("2. Document Ingestion")
#             uploaded_files = st.file_uploader(
#                 "Upload your PDF files",
#                 type=["pdf"],
#                 accept_multiple_files=True
#             )

#             if st.button("Embed & Store Documents", disabled=not uploaded_files):
#                 with st.spinner("Processing documents... This may take a moment."):
#                     ingest_documents(uploaded_files, client, embeddings, qdrant_url, qdrant_api_key)

#             if st.session_state.is_ingested:
#                 st.success("✅ Data is ready for chat!")
#             else:
#                 st.warning("Upload PDFs and click 'Embed & Store Documents' to begin chatting.")

#             # Evaluation Settings
#             st.header("3. Evaluation Settings")
#             st.session_state.enable_evaluation = st.checkbox(
#                 "Enable Response Evaluation",
#                 value=st.session_state.enable_evaluation,
#                 help="Evaluate each response for quality (uses extra LLM calls)"
#             )
            
#             if st.button("📊 View Evaluation Report"):
#                 if os.path.exists("rag_evaluations.json"):
#                     with open("rag_evaluations.json", 'r') as f:
#                         evals = json.load(f)
                    
#                     if evals:
#                         avg_relevance = sum(e["scores"]["relevance"] for e in evals) / len(evals)
#                         avg_faithfulness = sum(e["scores"]["faithfulness"] for e in evals) / len(evals)
#                         avg_completeness = sum(e["scores"]["completeness"] for e in evals) / len(evals)
#                         avg_similarity = sum(e["scores"].get("context_similarity", 0) for e in evals) / len(evals)
                        
#                         st.success(f"""
# **Overall Metrics ({len(evals)} responses)**
# - Avg Relevance: {avg_relevance:.1f}/10
# - Avg Faithfulness: {avg_faithfulness:.1f}/10
# - Avg Completeness: {avg_completeness:.1f}/10
# - Avg Retrieval Accuracy: {avg_similarity:.1f}%
#                         """)
#                     else:
#                         st.info("No evaluations yet.")
#                 else:
#                     st.info("No evaluations file found.")

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

#         # Setup Retriever
#         qdrant_vectorstore = QdrantVectorStore(
#             client=client,
#             collection_name=COLLECTION_NAME,
#             embedding=embeddings
#         )

#         # Retrieve top 7 relevant documents with MMR for diversity
#         retriever = qdrant_vectorstore.as_retriever(
#             search_type="mmr",  # Maximum Marginal Relevance for diverse results
#             search_kwargs={
#                 "k": 7,  # Retrieve more documents
#                 "fetch_k": 20  # Consider more candidates
#             }
#         )

#         system_prompt = (
#             "You are an expert RAG assistant helping answer questions about the ADSAI curriculum. "
#             "Answer the user's question using ONLY the provided context. Be SPECIFIC and DETAILED. "
#             "If the context mentions specific courses, modules, tracks, requirements, or details, include them in your answer. "
#             "If the context does not contain the answer, state clearly that the information is not available. "
#             "Always cite source document name(s) and page number(s) at the end of your response. "
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

#                         final_output = f"{full_response}\n\n---\n**Sources Used:**\n{sources}"
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
#                                     embeddings=embeddings  # Pass embeddings
#                                 )
                                
#                                 st.markdown("### 📊 Response Quality Scores")
#                                 col1, col2, col3, col4 = st.columns(4)  # 4 columns now
                                
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
                                
#                                 with st.expander("📝 Evaluation Explanation"):
#                                     st.write(eval_scores.get("explanation", "No explanation provided"))
#                                     st.write(f"\n**Retrieval Accuracy**: Measures how semantically similar the retrieved context is to your question. Higher percentages mean better document retrieval.")
                                
#                                 # Save evaluation
#                                 source_list = [f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" 
#                                               for doc in source_docs]
#                                 save_evaluation(prompt, full_response, source_list, eval_scores)

#                     except Exception as e:
#                         st.error(f"An error occurred during chat: {e}")
#                         st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
        
#         except Exception as e:
#             st.warning("Falling back to manual retrieve -> prompt -> LLM.")
#             try:
#                 docs = retriever.get_relevant_documents(prompt)
#                 context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
                
#                 composed_prompt = (
#                     "You are an expert RAG assistant. Answer the user's question only based on the provided context.\n"
#                     "If the context does not contain the answer, state clearly that the information is not available.\n"
#                     f"Context:\n{context_text}\n\nUser question: {prompt}\n\nAnswer:"
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