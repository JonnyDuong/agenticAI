import os
import ssl
import certifi
import json
import requests
import numpy as np
import faiss
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Environment and SSL Setup
# -----------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["CURL_CA_BUNDLE"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file")

MODEL = "mistralai/mixtral-8x7b-instruct"

# -----------------------------
# Load FAISS index + chunks
# -----------------------------
base_dir = Path(__file__).parent
embedding_path = base_dir / "RAG" / "embeddings" / "mistral_index_sentences" / "index.faiss"
chunk_lookup_path = base_dir / "RAG" / "mistral_chunk_lookup.txt"

logging.info(f"Base directory: {base_dir}")
logging.info(f"Looking for embedding index at: {embedding_path}")
logging.info(f"Looking for chunk lookup at: {chunk_lookup_path}")

if not embedding_path.exists():
    raise FileNotFoundError(f"Index not found at: {embedding_path}")
logging.info("Loading FAISS index...")
faiss_index = faiss.read_index(str(embedding_path))
logging.info(f"FAISS index loaded with {faiss_index.ntotal} vectors")

if not chunk_lookup_path.exists():
    raise FileNotFoundError(f"Chunk file not found at: {chunk_lookup_path}")
logging.info("Loading text chunks...")
with open(chunk_lookup_path, encoding="utf-8") as f:
    chunks = f.read().split("--- Chunk ")[1:]
    chunks = [chunk.split("\n", 1)[1].strip() for chunk in chunks]
logging.info(f"Loaded {len(chunks)} text chunks")

logging.info("Using remote embedding model")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

logging.info("Using online HuggingFace embedding model")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# -----------------------------
# Tool: FAISS-based RAG Retrieval
# -----------------------------
def retrieve_relevant_context(query: str) -> str:
    logging.info(f"RAG query: {query}")
    try:
        # Embed the query using the same embedding model as in preprocess.py
        query_vec = embedding_model.embed_query(query)
        
        # Convert to numpy array and ensure correct dtype (float32)
        query_vec = np.array([query_vec], dtype=np.float32)
        
        # Normalize the vector to unit length (L2 norm=1) - same as in FAISS indexing
        faiss.normalize_L2(query_vec)

        logging.info(f"Running FAISS search with embedding of shape {query_vec.shape}")
        
        # Retrieve more results to ensure we have enough valid ones
        top_k = 10  # Get more results to filter from
        D, I = faiss_index.search(query_vec, top_k)
        logging.info(f"FAISS search returned indices: {I[0]} with distances: {D[0]}")
        
        # Filter out indices that are out of range
        valid_indices_with_scores = [(idx, score) for idx, score in zip(I[0], D[0]) if 0 <= idx < len(chunks)]
        if len(valid_indices_with_scores) < len(I[0]):
            logging.warning(f"Filtered out {len(I[0]) - len(valid_indices_with_scores)} out-of-range indices. FAISS index may be out of sync with chunks.")
        
        # Only keep results with a good similarity score (lower distance is better)
        similarity_threshold = 1.2  # Adjust this threshold as needed
        filtered_indices = [(idx, score) for idx, score in valid_indices_with_scores if score < similarity_threshold]
        
        # Take the top 5 most relevant chunks
        filtered_indices = filtered_indices[:5]
        
        # Get the chunks for the valid indices
        top_chunks = [chunks[idx] for idx, _ in filtered_indices]
        logging.info(f"Retrieved {len(top_chunks)} chunks")
        
        # Return a helpful message if no valid chunks were found
        if not top_chunks:
            return "No relevant information found in the knowledge base."
        
        # Add a header to indicate this is retrieved content
        result = "Retrieved content:\n" + "\n\n".join(top_chunks)
        return result
    except Exception as e:
        logging.error(f"Error in RAG retrieval: {e}", exc_info=True)
        return f"Error retrieving context: {str(e)}"

rag_tool = Tool(
    name="RAGRetriever",
    func=retrieve_relevant_context,
    description="Use this tool to retrieve background information from Magistral documents using semantic vector search."
)

# -----------------------------
# Custom OpenRouter ChatModel Wrapper
# -----------------------------
class OpenRouterChat(BaseChatModel):
    def _llm_type(self) -> str:
        return "openrouter"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "LangChain Agent via OpenRouter",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": [{"role": self._convert_role(m.type), "content": m.content} for m in messages],
        }

        logging.info(f"Sending request to OpenRouter API with model: {MODEL}")
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            logging.info(f"OpenRouter API response status: {response.status_code}")
            response.raise_for_status()

            data = response.json()
            logging.info("Successfully parsed API response")
            answer = data["choices"][0]["message"]["content"]
            
            message = AIMessage(content=answer)
            chat_generation = ChatGeneration(message=message)
            return ChatResult(generations=[chat_generation])
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {e}")
            if hasattr(e, 'response') and e.response:
                logging.error(f"Response content: {e.response.text}")
            raise

    @staticmethod
    def _convert_role(message_type):
        if message_type == "ai":
            return "assistant"
        elif message_type == "human":
            return "user"
        elif message_type == "system":
            return "system"
        return message_type

# -----------------------------
# Create Flask API
# -----------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS  # For handling CORS (Cross Origin Resource Sharing)

# Initialize the OpenRouterChat and Agent
llm = OpenRouterChat()
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return app.send_static_file('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """
    Process a query through the RAG + OpenRouter LLM agent
    
    Request body:
    {
        "query": "Your query here"
    }
    
    Response:
    {
        "response": "Agent's response",
        "status": "success" or "error"
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Missing 'query' in request"}), 400
        
        query = data['query']
        logging.info(f"Received query: {query}")
        
        response = agent.run(query)
        
        return jsonify({
            "status": "success",
            "response": response
        })
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Simple test endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

# Run the Flask app if executed directly
if __name__ == '__main__':
    # Default test for debugging
    # query = "How does Magistral use reinforcement learning?"
    # response = agent.run(query)
    # print("\n🧠 Final Answer from OpenRouter Agent:\n")
    # print(response)
    
    # Start the API server
    print("Starting LLM API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
