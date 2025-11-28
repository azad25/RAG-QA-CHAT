# RAG based Q/A AI Chat using Langchain, FastAPI, Qdrant

A Retrieval-Augmented Generation (RAG) service for intelligent question-answering about real estate properties, services, and policies. Built with FastAPI, LangChain, Qdrant, and Hugging Face models.

## Features

- 🔍 **Semantic Search**: Finds relevant information from your documents using vector embeddings
- 🤖 **AI-Powered Responses**: Generates natural language answers using Hugging Face LLM models
- 📚 **Multi-Document Support**: Indexes multiple document types (listings, services, pricing, terms)
- 🎯 **Source Attribution**: Returns answer with source documents and relevance scores
- ⚡ **Fast & Scalable**: Built with FastAPI and Qdrant vector database

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   FastAPI   │─────▶│  RAG Engine  │─────▶│   Qdrant    │
│   Service   │      │  (LangChain) │      │  Vector DB  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Hugging Face │
                     │  LLM Model   │
                     └──────────────┘
```

## Tech Stack

- **API Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **Vector Database**: Qdrant
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM Model**: Hugging Face Router (`moonshotai/Kimi-K2-Instruct-0905`)
- **Language**: Python 3.10

## Prerequisites

- Docker & Docker Compose
- Hugging Face API Token ([Get one here](https://huggingface.co))

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/azad25/RAG-QA-CHAT
cd RAG-QA-CHAT
```

### 2. Configure Environment Variables

Create a `.env` file or update `config.py`:

```python
# Qdrant Configuration
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "xyz_real_estate"

# Embedding Model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Document Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Hugging Face Configuration
HF_TOKEN = "your_huggingface_token_here"
LLAMA_MODEL = "moonshotai/Kimi-K2-Instruct-0905"
```

### 3. Prepare Your Documents

Add your documents to the `data/` directory:

```bash
data/
├── listing.txt      # Property listings
├── services.txt     # Service offerings
├── pricing.txt      # Pricing information
└── terms.txt        # Terms and conditions
```

**Note**: Only `.txt` files are currently supported.

### 4. Start the Services

```bash
docker-compose up --build
```

The service will:
- Start Qdrant on port 6333
- Start FastAPI on port 8000
- Load documents from `data/` directory
- Create vector embeddings and store in Qdrant

## API Usage

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Ask a Question (Simple)

**Endpoint**: `POST /ask`

**Request**:
```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Show me properties in Los Angeles"
}'
```

**Response**:
```json
{
  "question": "Show me properties in Los Angeles",
  "answer": "Here are the available properties in Los Angeles:\n\n**1. West Hollywood Condo (CA-1)**\n- **Price:** $1,495,000\n- **Bed/Bath:** 2 Bed / 2 Bath\n- **Features:** Corner unit, concierge, rooftop pool, 2-car garage\n- **Timeline:** Immediate Close\n\n**2. Studio City Single Family Home (CA-2)**\n- **Price:** $2,150,000\n- **Bed/Bath:** 4 Bed / 3.5 Bath\n- **Features:** Expansive backyard, private office, newly updated kitchen, great school district\n- **Timeline:** 60-90 Days\n\n**3. Silver Lake Townhouse (CA-3)**\n- **Price:** $885,000\n- **Bed/Bath:** 3 Bed / 2.5 Bath\n- **Features:** Walkable to shops/transit, private patio, low HOA, ideal for commuters\n- **Timeline:** Flexible"
}
```

#### 2. Ask with Sources (Detailed)

**Endpoint**: `POST /ask-with-sources`

**Request**:
```bash
curl -X 'POST' \
  'http://localhost:8000/ask-with-sources' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What are your commission rates?"
}'
```

**Response**:
```json
{
  "question": "What are your commission rates?",
  "answer": "Our commission rates vary by market...",
  "sources": [
    {
      "content": "A. Los Angeles & New York (High-Value, High-Service Markets)...",
      "metadata": {
        "source": "data/pricing.txt"
      },
      "score": 0.85
    }
  ]
}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## Project Structure

```
.
├── main.py                 # FastAPI application
├── rag_engine.py          # RAG logic with LangChain & Qdrant
├── models.py              # Pydantic models
├── config.py              # Configuration settings
├── data/                  # Document directory
│   ├── listing.txt
│   ├── services.txt
│   ├── pricing.txt
│   └── terms.txt
├── docker-compose.yml     # Docker services configuration
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration Options

### Embedding Model
Change the embedding model in `config.py`:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384 dimensions
# or
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality, 768 dimensions
```

**Note**: If you change the embedding model, update the vector size in `rag_engine.py`:
```python
vectors_config=VectorParams(
    size=384,  # Change to 768 for mpnet-base-v2
    distance=Distance.COSINE
)
```

### LLM Model
Change the language model in `config.py`:
```python
LLAMA_MODEL = "moonshotai/Kimi-K2-Instruct-0905"
# or
LLAMA_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# or
LLAMA_MODEL = "google/flan-t5-large"
```

### Chunk Size & Overlap
Adjust document chunking in `config.py`:
```python
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 50    # Overlap between chunks
```

### Retrieval Settings
Change number of documents retrieved:
```python
TOP_K = 3  # Number of similar documents to retrieve
```

## Troubleshooting

### Collection Already Exists

If you need to recreate the collection with fresh data, set in `rag_engine.py`:
```python
FORCE_RECREATE = True  # In _init_collection method
```

Then restart:
```bash
docker-compose down
docker-compose up --build
```

### Documents Not Loading

Check the logs:
```bash
docker-compose logs rag-service
```

Ensure:
- Files are in `data/` directory
- Files have `.txt` extension
- Files are UTF-8 encoded

### LLM Errors

If you get model errors, verify:
- Your Hugging Face token is valid
- The model name is correct
- You have internet access for API calls

## Performance Optimization

### For Production

1. **Use a larger embedding model** for better accuracy:
   ```python
   EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
   ```

2. **Increase TOP_K** for more context:
   ```python
   TOP_K = 5
   ```

3. **Add caching** for repeated queries

4. **Use a dedicated LLM** (OpenAI, Anthropic) instead of Hugging Face Inference API

## Development

### Install Dependencies Locally

```bash
pip install -r requirements.txt
```

### Run Without Docker

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Update config.py
QDRANT_HOST = "localhost"

# Run the service
uvicorn main:app --reload
```

## Dependencies

```txt
fastapi
uvicorn[standard]
langchain-core
langchain-community
langchain-openai
qdrant-client
sentence-transformers
huggingface-hub
pydantic
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.