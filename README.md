# Multi-Document AI Research Assistant (RAG)

This project is a Generative AI application that allows users to upload multiple PDF documents and ask questions based on their content using Retrieval-Augmented Generation (RAG).

---

## Overview

The system combines large language models with vector search and document retrieval to generate accurate, context-aware answers grounded in uploaded documents.

---

## Features

* Upload and process multiple PDF documents
* Ask questions in natural language
* Context-aware responses using conversation memory
* Answers generated strictly from document content
* Fast retrieval using vector database
* Reduced hallucination through controlled prompting

---

## Tech Stack

* Frontend: Streamlit
* Backend: Python
* LLM: Groq (LLaMA 3)
* Framework: LangChain
* Embeddings: HuggingFace
* Vector Database: ChromaDB

---

## How It Works

1. Upload PDF documents
2. Extract and split text into chunks
3. Convert chunks into embeddings
4. Store embeddings in a vector database
5. Convert user query into embedding
6. Retrieve relevant document chunks
7. Generate answer using LLM and retrieved context

---

## Live Demo

https://multi-document-ai-assistant-kqee4nxdtasrm2vurxtkla.streamlit.app/

---

## Installation & Setup

```bash
git clone https://github.com/sahilkumar01git/multi-document-ai-assistant.git
cd multi-document-ai-assistant
python -m pip install -r requirements.txt
```

---

## API Key Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Run Locally

```bash
python -m streamlit run main.py
```

---

## Project Structure

```
main.py
requirements.txt
README.md
screenshot.png
sample.pdf
```

---

## Use Cases

* Research assistance
* Document-based question answering
* Internal knowledge systems
* Customer support automation

---

## Future Improvements

* Source highlighting (file and page reference)
* Multi-document comparison
* Improved user interface
* Authentication and user sessions

---

## Note

This project demonstrates a practical implementation of Retrieval-Augmented Generation (RAG) using modern large language models and vector search.
