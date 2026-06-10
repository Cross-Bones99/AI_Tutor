import os

from tavily import TavilyClient

from rag_chain import ask_question


client=TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)




def retrieval_router(state):

    question = state["question"]

    vectorstore = state["vectorstore"]

    results = vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    best_score = results[0][1]

    print("Best Score:", best_score)

    if best_score < 1.2:
        route = "rag"
    else:
        route = "web"

    return {
        **state,
        "route": route
    }



def rag_node(state):

    answer,docs = ask_question(
        state["question"],
        state["vectorstore"],
        state.get("history","")
    )

    return {
        **state,
        "answer": answer,
        "sources":docs
    }




def web_node(state):

    query = state["question"]

    results = client.search(
        query=query,
        max_results=5
    )

    context = "\n".join(
        r["content"]
        for r in results["results"]
    )

    prompt = f"""
    Answer the question using the
    provided web search context.

    Context:
    {context}

    Question:
    {query}
    """

    response = state["llm"].invoke(
        prompt
    )

    return {
        **state,
        "answer": response.content
    }