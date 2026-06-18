from utils.vector_store import get_vectorstore





def get_context(
    vectorstore,
    query,
    k=5
):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context, docs