# Multi-Document AI Assistant

A Streamlit chatbot that answers normal questions with Groq and switches to grounded RAG when PDF files are uploaded.

## Features

- Upload multiple PDFs in one session
- ChatGPT-style conversation with right-aligned user messages and left-aligned assistant replies
- General chat mode without requiring a PDF
- Streaming responses with a visible thinking state
- Groq `openai/gpt-oss-120b` for answer generation
- Local Hugging Face embeddings and Chroma retrieval
- Answers grounded in retrieved document context when PDFs are present
- Source document and page citations
- Cached embeddings and document processing
- Isolated vector collections for each document set
- MMR retrieval for more diverse and relevant chunks
- Friendly handling for missing keys, unreadable PDFs, and API errors

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Groq key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The app also accepts `groq_api_key` for compatibility with an existing local `.env` file. Never commit `.env` or expose the key in source code.

## Run

```powershell
python -m streamlit run main.py
```

Then open the local URL shown by Streamlit.

## Tests

```powershell
pytest -q
```

## Architecture

1. Streamlit accepts normal chat questions and optional PDF uploads.
2. Without PDFs, Groq answers as a general-purpose conversational assistant.
3. With PDFs, files are written to temporary paths only while `PyPDFLoader` reads them.
4. Text is split into overlapping chunks and embedded locally with normalized vectors.
5. Chroma stores the chunks in a collection derived from the uploaded content.
6. A history-aware MMR retriever resolves follow-up questions and selects diverse evidence.
7. Groq receives the question and retrieved context, then streams a grounded answer.
8. Retrieved source pages are shown below document-based answers.

## Project Structure

```text
main.py                 Streamlit entry point
app/
	config.py             Environment variables and application constants
	services/
		chat_service.py     General Groq chat and message conversion
		rag_service.py      PDF loading, embeddings, Chroma, and retrieval chains
	ui/
		app.py              Streamlit page orchestration
		sidebar.py          Upload and conversation controls
		styles.py           Shared visual styling and empty state
tests/                  Automated tests
```

The service modules are intentionally in-process for this Streamlit deployment. They provide clean boundaries so chat, retrieval, and UI can later be extracted into independently deployed services if the project grows.

## Deployment

For Streamlit Community Cloud, add `GROQ_API_KEY` under the app's secrets settings instead of committing a `.env` file.
