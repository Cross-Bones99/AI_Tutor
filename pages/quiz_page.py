import streamlit as st

from agents.graph import graph
from rag_chain import llm


def show_quiz_page():

    st.header("📝 Quiz Generator")

    topic = st.text_input(
        "Topic"
    )

    num_questions = st.slider(
        "Questions",
        5,
        20,
        10
    )

    if st.button(
        "Generate Quiz"
    ):

        query = (
            f"Generate {num_questions} "
            f"MCQs on {topic}"
        )

        result = graph.invoke(
            {
                "question": query,
                "route": "",
                "answer": "",
                "vectorstore": st.session_state.vectorstore,
                "llm": llm,
                "history": ""
            }
        )

        st.markdown(
            result["answer"]
        )