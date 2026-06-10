from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

# LLM
llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


prompt = PromptTemplate(
    input_variables=["context", "question","history"],
    template="""
You are an AI Tutor.

Use ONLY the provided context.

Conversation History:
{history}

Document Context:
{context}

Current Question:
{question}

Instructions:

- Explain clearly.
- Use examples when useful.
- Answer follow-up questions using the conversation history.
- If the answer is not in the context, say so.

Answer:
"""
)

def ask_question(question,vectorstore,history=""):
    
    retriever = get_retriever(vectorstore)

    docs=retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.format(
        context=context,
        question=question,
        history=history
    )

    response = llm.invoke(final_prompt)

    return response.content, docs



def stream_answer(question, vectorstore,chat_history):


    history_text = ""

    for msg in chat_history[-6:]:

        role = msg["role"]

        content = msg["content"]

        history_text += (
            f"{role}: {content}\n"
        )
    





    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5},
        search_type="mmr"
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.format(
        context=context,
        question=question,
        history=history_text
    )

    return llm.stream(final_prompt), docs