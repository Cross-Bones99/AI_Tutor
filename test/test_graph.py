from agents.graph import graph
from rag_chain import llm
from utils.vector_store import create_vectorstore_from_pdfs

vectorstore = create_vectorstore_from_pdfs(
    ["data/Attention.pdf"]
)

result = graph.invoke(
    {
        "question": "What is machine learning?",
        "vectorstore": vectorstore,
        "llm": llm,
        "route": "",
        "answer": ""
    }
)

print(result)