import os

from tavily import TavilyClient

from rag_chain import ask_question
from utils.shared_retrievel import get_context


client=TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)




def retrieval_router(state):

    question = state["question"].lower()



    study_keywords = [
    "study plan",
    "roadmap",
    "learning plan",
    "schedule"
    ]


    if any(
        keyword in question
        for  keyword in study_keywords

    ):
        return {
            **state,
            "route": "study"
        }

    quiz_keywords = [
        "quiz",
        "mcq",
        "multiple choice",
        "test me",
        "practice questions"
    ]

    if any(
        keyword in question
        for keyword in quiz_keywords
    ):
        return {
            **state,
            "route": "quiz"
        }

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





def quiz_node(state):

    context, docs = get_context(
        state["vectorstore"],
        state["question"]
    )

    prompt = f"""
    You are an educational quiz generator.

    Generate 10 MCQs ONLY from the provided notes.

    Notes:
    {context}

    Requirements:

    - Four options
    - Mark correct answer
    - Cover different concepts
    - Do not invent information

    Format:

    Question 1
    A)
    B)
    C)
    D)

    Correct Answer:
    """

    response = state["llm"].invoke(prompt)

    return {
        **state,
        "answer": response.content,
        "sources": docs
    }



def study_plan_node(state):

    context, docs = get_context(
        state["vectorstore"],
        state["question"]
    )

    prompt = f"""
    You are an expert learning coach.

    Create a study plan ONLY using
    the supplied notes.

    Notes:
    {context}

    Requirements:

    - Identify key topics
    - Arrange logically
    - Include revision days
    - Include practice sessions

    Generate a structured plan.
    """

    response = state["llm"].invoke(prompt)

    return {
        **state,
        "answer": response.content,
        "sources": docs
    }