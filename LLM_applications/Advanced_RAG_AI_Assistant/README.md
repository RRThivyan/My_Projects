# Uttar Pradesh Government RAG AI Assistant

This project is a Retrieval-Augmented Generation (RAG) AI assistant that provides information about the Uttar Pradesh government. It can be run as a FastAPI application or as a command-line interface.

## Features

- **RAG-based Chatbot:** The assistant uses a RAG model to answer questions about the Uttar Pradesh government.
- **FastAPI Interface:** The application can be run as a FastAPI server, with a `/chat` endpoint for interacting with the chatbot.
- **Command-Line Interface:** The application can also be run as a command-line interface, allowing you to chat with the assistant in your terminal.
- **Query Expansion:** The assistant uses various techniques to expand user queries, including synonyms, abbreviations, and alternate titles.
- **Reranking:** The assistant uses a reranker model to improve the relevance of the search results.
- **Multi-lingual Support:** The assistant can handle queries in both English and Hindi.

## Project Structure

- **`main.py`:** The main script for the FastAPI application, containing the core RAG logic and the FastAPI server.
- **`processing.py`:** The core script where the RAG logic was developed and tested. It can be run as a command-line interface.
- **`requirements.txt`:** A file containing all the Python packages required to run the project.
- **`.env`:** A file for storing environment variables, such as API keys and endpoints.
- **`embedding/`:** A directory containing the script and data for generating the embeddings.
  - **`embed.py`:** A script that extracts text from a PDF file, chunks the text, generates embeddings, and saves them to a FAISS index.
  - **`final_data.pdf`:** The PDF file used to generate the embeddings.
- **`faiss_index_single_pdf/`:** A directory containing the FAISS index.
- **`bm25_index.pkl`:** The BM25 index.
- **`documents.pkl`:** The documents used for the BM25 index.
- **`chat_history.json`:** A file for storing the chat history.

## Getting Started

### Prerequisites

- Python 3.7+
- An OpenAI API key and endpoint

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your API key and endpoint:
   ```
   GPT4_API_KEY=your-api-key
   GPT4_ENDPOINT=your-endpoint
   ```

### Dependencies

The main dependencies are:
- `numpy`
- `faiss-cpu`
- `openai`
- `langchain`
- `sentence-transformers`
- `fastapi`
- `uvicorn`
- `python-dotenv`
- `PyPDF2`
- `torch`
- `transformers`
- `langdetect`
- `rapidfuzz`
- `rank_bm25`

### Usage

#### FastAPI

To run the application as a FastAPI server, run the following command:

```bash
python main.py
```

You can then send a POST request to the `/chat` endpoint with a JSON payload containing the user's query:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"user_query": "Who is the current chief minister of Uttar Pradesh?", "session_id": null}'
```

#### Command-Line Interface

To run the application as a command-line interface, run the following command:

```bash
python processing.py
```

You can then chat with the assistant in your terminal.
