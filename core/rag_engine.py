import os
import tempfile
from typing import Tuple, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from core import config

def format_docs_with_sources(docs) -> Tuple[str, List[dict]]:
    """Formats retrieved documents into context string and extracts citations."""
    formatted_chunks = []
    citations = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", 0) + 1  # 1-indexed page
        source_name = os.path.basename(doc.metadata.get("source", "PDF Document"))
        citations.append({
            "chunk_id": i,
            "page": page,
            "snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
            "source": source_name
        })
        formatted_chunks.append(f"[Excerpt {i} | Page {page}]:\n{doc.page_content}")
    
    return "\n\n".join(formatted_chunks), citations

def initialize_rag_from_upload(uploaded_file, api_key: str):
    """
    Processes an uploaded PDF file:
    - Writes to a temporary file
    - Extracts and chunks content
    - Generates embeddings and builds Chroma in-memory index
    - Constructs and returns the retriever and LLM chain
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.DEFAULT_EMBEDDING_MODEL,
            google_api_key=api_key
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.DEFAULT_TOP_K}
        )

        llm = ChatGoogleGenerativeAI(
            model=config.DEFAULT_LLM_MODEL,
            temperature=0.0,
            google_api_key=api_key
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI document assistant answering user questions based strictly on the provided context.\n"
                "Ground rules:\n"
                "1. Answer ONLY using the facts from the retrieved passages below.\n"
                "2. When answering, specify which page(s) or excerpt numbers contained the information.\n"
                "3. If the context does not provide sufficient details to answer, state: "
                "'I cannot find the answer in the provided document.'\n\n"
                "Context:\n{context}"
            )),
            ("human", "{question}")
        ])

        generation_chain = prompt | llm | StrOutputParser()

        return {
            "retriever": retriever,
            "chain": generation_chain,
            "total_pages": len(documents),
            "total_chunks": len(chunks)
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
