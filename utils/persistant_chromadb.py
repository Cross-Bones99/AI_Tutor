from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vectorstore():

    return Chroma(
        collection_name="ai_tutor",
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )


vectorstore = get_vectorstore()


