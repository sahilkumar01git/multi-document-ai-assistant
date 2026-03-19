from dotenv import load_dotenv
import os
import tempfile

from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


load_dotenv()

st.title("Multi-Document AI Assistant")
st.sidebar.title("PDFs")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=True
)

query = st.text_input("Ask a question:")

st.session_state.memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

@st.cache_resource
def process_documents(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordata = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordata


if uploaded_file:
    vectordata = process_documents(uploaded_file)
    retriever = vectordata.as_retriever(search_kwargs={"k": 3})

    groq_LLM = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=500,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question only from the given context.
If the answer is not present in the context, say:
"I could not find that in the uploaded documents."

Context:
{context}

Question:
{input}
Answer:
""")

    document_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_LLM,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer"
    )

    if query:
        response = document_chain.invoke({"question": query})
        st.subheader("Answer")
        st.write(response["answer"])
