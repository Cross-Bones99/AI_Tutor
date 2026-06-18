import os
import streamlit as st

from pages.chat_page import show_chat_page
from pages.quiz_page import show_quiz_page
from pages.study_planner import show_study_planner

from utils.vector_store import (
    get_vectorstore,
    add_pdf_to_vectorstore
)

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="AI Tutor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Tutor")
st.markdown(
    "An AI-powered learning platform with Chat, Quiz Generation, and Study Planning."
)

# ----------------------------------
# Session State
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

# ----------------------------------
# Sidebar - Knowledge Base
# ----------------------------------

st.sidebar.header("📚 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    for uploaded_file in uploaded_files:

        pdf_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        added = add_pdf_to_vectorstore(
            pdf_path
        )

        if added:

            st.sidebar.success(
                f"✅ {uploaded_file.name} indexed"
            )

        else:

            st.sidebar.warning(
                f"⚠️ {uploaded_file.name} already exists"
            )

# ----------------------------------
# Display Indexed Documents
# ----------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "Indexed Documents"
)

try:

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

    if files:

        for file in sorted(files):

            st.sidebar.write(
                f"📄 {file}"
            )

        st.sidebar.success(
            f"Total PDFs: {len(files)}"
        )

    else:

        st.sidebar.info(
            "No PDFs indexed yet."
        )

except Exception:

    st.sidebar.info(
        "No documents available."
    )

# ----------------------------------
# Navigation
# ----------------------------------

st.sidebar.divider()

page = st.sidebar.selectbox(
    "Choose Feature",
    [
        "Chat Assistant",
        "Quiz Generator",
        "Study Planner"
    ]
)

# ----------------------------------
# Route Pages
# ----------------------------------

if page == "Chat Assistant":

    show_chat_page()

elif page == "Quiz Generator":

    show_quiz_page()

elif page == "Study Planner":

    show_study_planner()