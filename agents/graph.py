from langgraph.graph import StateGraph,START,END

from agents.state import AgentState
from agents.nodes import retrieval_router, rag_node, web_node,quiz_node,study_plan_node


# Conditional Edge

def route_decision(state):

    return state["route"]



builder=StateGraph(AgentState)

builder.add_node("router", retrieval_router)
builder.add_node("rag", rag_node)
builder.add_node("web", web_node)
builder.add_node("quiz", quiz_node)
builder.add_node("study",study_plan_node)

builder.add_edge(START,"router")

builder.add_conditional_edges("router",route_decision,{
    "rag":"rag",
    "web":"web",
    "quiz":"quiz",
    "study":"study"
})


builder.add_edge(
    "rag",
    END
)

builder.add_edge(
    "web",
    END
)


builder.add_edge(
    "quiz",
    END
)

builder.add_edge(
    "study",
    END
)

graph = builder.compile()



