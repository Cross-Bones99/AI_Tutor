import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_vectorstore():

    return Chroma(
        collection_name="ai_tutor",
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )


def create_vectorstore_from_pdfs(pdf_paths):

    all_docs = []

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        filename = os.path.basename(pdf_path)

        for doc in docs:
            doc.metadata["source_file"] = filename

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    vectorstore = get_vectorstore()

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids

    )

    return vectorstore




def add_pdf_to_vectorstore(pdf_path):

    vectorstore = get_vectorstore()

    filename = os.path.basename(pdf_path)

    existing = vectorstore.get(
        where={"source_file": filename}
    )

    if existing["ids"]:
        return False

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    return True