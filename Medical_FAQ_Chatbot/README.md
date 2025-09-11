# RAG-based Medical FAQ Chatbot

This project is a simple yet powerful chatbot that uses the Retrieval-Augmented Generation (RAG) architecture to answer medical questions. It leverages a dataset of medical FAQs, OpenAI's language models, and a FAISS vector store to provide accurate and contextually relevant answers.

## Features

- **RAG Pipeline**: The chatbot uses a RAG pipeline to retrieve relevant information from a knowledge base and generate human-like answers.
- **Streamlit Interface**: A simple and intuitive web interface built with Streamlit allows users to interact with the chatbot.
- **OpenAI Integration**: The project uses OpenAI's powerful language models for text generation and Huggingface sentence transformers for embeddings.
- **FAISS Vector Store**: A FAISS vector store is used for efficient similarity search and retrieval of medical information.

## Project Structure

```
Medical_FAQ_Chatbot/
├── .env
├── app.py
├── create_vectorstore.py
├── faiss_index/
├── requirements.txt
└── README.md
```

- **`.env`**: Stores the OpenAI API key and endpoint.
- **`app.py`**: The main Streamlit application file.
- **`create_vectorstore.py`**: A script to create the FAISS vector store from the medical FAQ dataset.
- **`faiss_index/`**: The directory where the FAISS vector store is saved.
- **`requirements.txt`**: A list of all the Python libraries required to run the project.
- **`README.md`**: This file.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Medical_FAQ_Chatbot
    ```

2.  **Create a virtual environment and install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up your OpenAI API key**:
    -   Create a `.env` file in the project root.
    -   Add your OpenAI API key and endpoint to the `.env` file:
        ```
        OPENAI_API_ENDPOINT="your-api-endpoint"
        OPENAI_API_KEY="your-api-key"
        ```

4.  **Create the vector store**:
    -   Make sure the `medicalqa.csv` file is in the parent directory.
    -   Run the `create_vectorstore.py` script:
        ```bash
        python create_vectorstore.py
        ```

## How to Run the Application

Once you have completed the setup, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will open the chatbot interface in your web browser. You can then start asking medical questions.

## Design Choices

-   **Streamlit**: Chosen for its simplicity and ease of use in creating interactive web applications.
-   **FAISS**: A lightweight and efficient library for similarity search, making it a good choice for this project.
-   **OpenAI**: Provides high-quality language models that are well-suited for RAG applications.
-   **`.env` for API Keys**: This is a standard practice for securely managing API keys and other sensitive information.
