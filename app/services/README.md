Service layer

- `chat_service.py` owns general Groq chat behavior.
- `rag_service.py` owns PDF loading, chunking, embeddings, retrieval, and document-grounded chains.

These are in-process service modules today. They can be extracted behind APIs later if the application needs independently deployed services.
