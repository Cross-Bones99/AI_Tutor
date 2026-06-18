import streamlit as st

from agents.graph import graph
from rag_chain import llm


def show_chat_page():

    st.header("📚 AI Tutor Chat")

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Ask a question..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            history = ""

            for msg in st.session_state.messages[-6:]:

                history += (
                    f"{msg['role']}: "
                    f"{msg['content']}\n"
                )

            result = graph.invoke(
                {
                    "question": prompt,
                    "route": "",
                    "answer": "",
                    "vectorstore": st.session_state.vectorstore,
                    "llm": llm,
                    "history": history
                }
            )

            st.markdown(
                result["answer"]
            )

            st.caption(
                f"🧭 Route Used: "
                f"{result['route'].upper()}"
            )

            if (
                result["route"] == "rag"
                and "sources" in result
            ):

                with st.expander(
                    "📄 Sources"
                ):

                    for doc in result["sources"]:

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        source_file = doc.metadata.get(
                            "source_file",
                            "Unknown"
                        )

                        st.markdown(
                            f"**File:** {source_file}"
                        )

                        st.markdown(
                            f"**Page:** {page + 1}"
                        )

                        st.write(
                            doc.page_content[:500]
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )