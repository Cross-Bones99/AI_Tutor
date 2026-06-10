

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0
)

response = llm.invoke(
    "Explain what Retrieval Augmented Generation is."
)

print(response.content)

