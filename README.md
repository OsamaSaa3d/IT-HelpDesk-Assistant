# Ticket Support System

RAG system for IT support tickets with semantic search and AI recommendations. Built with FastAPI backend and a clean web interface.

## Features

- Semantic search using FAISS and sentence transformers
- AI-powered recommendations via Google Gemini
- Web UI for easy access
- Processes tickets from CSV, Excel, and JSON files
- REST API for integration

## Project Structure

```
.
├── src/                          # Core modules
│   ├── config.py                 # Configuration
│   ├── data_processing.py        # Data processing
│   ├── embeddings.py             # Embedding model
│   ├── faiss_index.py            # FAISS index management
│   └── llm_client.py             # Gemini LLM client
│
├── Data/
│   ├── old_tickets/              # Raw ticket files
│   └── old_tickets_processed/    # Processed data and index
│
├── prepare_data.py               # Step 1: Process tickets
├── build_index.py                # Step 2: Build search index
├── server.py                     # FastAPI server
├── client.html                   # Web interface
└── requirements.txt
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### 1. Prepare the data

```bash
python prepare_data.py
```

Reads tickets from `Data/old_tickets/` (CSV, Excel, JSON) and unifies them into a single dataset.

### 2. Build the search index

```bash
python build_index.py
```

Generates embeddings and creates the FAISS index for semantic search.

### 3. Start the server

```bash
python server.py
```

Server runs on `http://localhost:8000`

### 4. Open the web interface

Open `client.html` in your browser. You can:
- **Search**: Find similar historical tickets
- **Get AI Recommendations**: Get suggested solutions based on similar tickets

## API Endpoints

The FastAPI server provides two endpoints:

- `POST /search` - Semantic search for similar tickets
- `POST /recommend` - Get AI-powered recommendations

Both accept JSON with `query` (string) and `top_k` (int, default: 5).

## Tech Stack

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Search**: FAISS (Facebook AI Similarity Search)
- **LLM**: Google Gemini
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JS

## Requirements

- Python 3.8+
- ~2GB RAM for embeddings
- Internet connection (first run downloads the model)

## Notes

This is a case study project for Aleph Alpha. The system processes historical IT tickets and uses semantic search to find similar issues, then generates recommendations using an LLM.

