from typing import TypedDict
from typing import Any




class AgentState(TypedDict):

    question: str

    route: str

    answer: str

    vectorstore: Any

    llm: Any

    sources: Any

    history: str
