from rag_chain import ask_question

question = input("Ask a question: ")

answer, docs = ask_question(question)

print("\nANSWER\n")
print(answer)

print("\nSOURCES\n")

for i, doc in enumerate(docs, start=1):
    print(f"\nSource {i}")
    print(doc.page_content[:300])