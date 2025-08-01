# RAG LLM API with OpenRouter

This project implements a Retrieval-Augmented Generation (RAG) system using FAISS for vector search and OpenRouter for LLM capabilities. It provides an API that can be integrated with any frontend.

## Setup

1. Make sure you have the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. Run the preprocessing script to generate embeddings (if not already done):
   ```
   python preprocess.py
   ```

4. Run the API server:
   ```
   python LLM_API.py
   ```

5. The server will start on port 5000 by default.

## Testing the API

There are three ways to interact with the API:

### 1. Web Interface
Open a web browser and go to `http://localhost:5000/` to access the web interface.

### 2. Command-line Testing Tool
Use the provided test client:
```
python test_api_client.py "Your question here"
```

### 3. Direct API Calls
You can make direct HTTP requests to the API:

```
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does Magistral use reinforcement learning?"}'
```

## API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/query` - Main query endpoint
  - Request body: `{"query": "Your question here"}`
  - Response: `{"status": "success", "response": "Answer from the LLM"}`

## Integration with Frontend

To integrate with your own frontend:
1. Make POST requests to the `/api/query` endpoint with a JSON body containing the "query" field
2. Handle the JSON response and display the "response" field to the user
