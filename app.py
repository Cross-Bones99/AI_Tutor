import os
import streamlit as st

from rag_chain import ask_question
from utils.vector_store import create_vectorstore_from_pdfs
from rag_chain import stream_answer

from utils.vector_store import get_vectorstore
from utils.vector_store import add_pdf_to_vectorstore


from agents.graph import graph
from rag_chain import llm



# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="AI Tutor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Tutor")
st.markdown("Upload a PDF and ask questions about it.")

# ----------------------------------
# Session State
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

# ----------------------------------
# Cache Vector Store
# ----------------------------------


# ----------------------------------
# PDF Upload
# ----------------------------------

uploaded_files = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    current_files = tuple(
        sorted(file.name for file in uploaded_files)
    )

    os.makedirs("temp", exist_ok=True)

    pdf_paths = []

    for uploaded_file in uploaded_files:

        pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
        )

        os.makedirs(
                "uploads",
                exist_ok=True
            )

        with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        added = add_pdf_to_vectorstore(
                pdf_path
            )

        if added:
                st.success(
                    f"{uploaded_file.name} indexed"
                )
        else:
                st.warning(
                    f"{uploaded_file.name} already exists"
                )

        st.session_state.indexed_files = current_files

        st.success("PDF indexed successfully!")





# ----------------------------------
# Display Chat History
# ----------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------
# Chat Input
# ----------------------------------

prompt = st.chat_input(
    "Ask a question about the PDF..."
)

# ----------------------------------
# Process Query
# ----------------------------------

if prompt:

    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # User Message

        with st.sidebar:
            st.header("Knowledge Base")

            collection = (
        st.session_state.vectorstore.get()
        )

        files = set()

        for metadata in collection["metadatas"]:

            if metadata:
                files.add(
                    metadata.get(
                        "source_file",
                        "Unknown"
                    )
                )

        for file in sorted(files):

            st.write(
                f"📄 {file}"
            )

        st.write(
            f"Total PDFs: {len(files)}"
        )

            

        st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

        with st.chat_message("user"):
            st.markdown(prompt)



        # Assistant Response

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                history = ""

                for msg in st.session_state.messages[-6:]:

                    history += (
                        f"{msg['role']}: {msg['content']}\n"
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

            full_response = result["answer"]

            placeholder = st.empty()
            placeholder.markdown(full_response)

            st.caption(
                f"🧭 Route Used: {result['route'].upper()}"
            )


            if result["route"] == "rag":

                with st.expander("📄 Sources"):

                    for doc in result["sources"]:

                        page = doc.metadata.get("page", "Unknown")

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
            else:
                st.info("Answer generated from web search.")


        
        # Saves Chat History
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response
            }
        )


        