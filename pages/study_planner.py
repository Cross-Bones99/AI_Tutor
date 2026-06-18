import streamlit as st

from agents.graph import graph
from rag_chain import llm


def show_study_planner():

    st.header("📅 Study Planner")

    topic = st.text_input(
        "Learning Goal"
    )

    days = st.slider(
        "Days",
        3,
        30,
        7
    )

    

    if st.button(
        "Generate Plan"
    ):

        query = (
    f"Create a {days} day study plan "
    f"from the uploaded notes"
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