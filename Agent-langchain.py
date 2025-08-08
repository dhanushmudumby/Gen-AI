from typing import Dict, TypedDict
from langgraph.graph import StateGraph,START, END
import os
from IPython.display import Image, display
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")


class AgentState(TypedDict):
    """Define the state structure for the agent."""
    number1:int
    number2:int
    operation1:str
    operation2:str
    result:str
    result2:str
    number3:int
    number4:int

def add_node(state:AgentState) -> AgentState:
    """Function to compliment the agent."""
    state["result"] = f"{state['number1']} {state['operation1']} {state['number2']} = {state['number1'] + state['number2']}"
    return state

def sub_node(state:AgentState) -> AgentState:
    """Function to compliment the agent."""
    state["result"] = f"{state['number1']} {state['operation1']} {state['number2']} = {state['number1'] - state['number2']}"
    return state

def add_node2(state:AgentState) -> AgentState:
    """Function to compliment the agent."""
    state["result2"] = f"{state['number3']} {state['operation2']} {state['number3']} = {state['number3'] + state['number4']}"
    return state

def sub_node2(state:AgentState) -> AgentState:
    """Function to compliment the agent."""
    state["result2"] = f"{state['number3']} {state['operation2']} {state['number4']} = {state['number3'] + state['number4']}"
    return state

def decision_node2(state:AgentState) ->AgentState:
    """Decision node to set the agent's name."""
    if state["operation2"]=='+':
        return "addition_operation2"
    elif state["operation2"]=='-':
        return "subtraction_operation2"

def decision_node(state:AgentState) ->AgentState:
    """Decision node to set the agent's name."""
    if state["operation1"]=='+':
        return "addition_operation"
    elif state["operation1"]=='-':
        return "subtraction_operation"

graph = StateGraph(AgentState)

graph.add_node("add_node", add_node)
graph.add_node("sub_node", sub_node)
graph.add_node("add_node2", add_node2)
graph.add_node("sub_node2", sub_node2)
graph.add_node("router", lambda state:state)
graph.add_node("router2", lambda state:state)
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router", 
    decision_node,
    {
        # Edge: Node format
        "addition_operation": "add_node",
        "subtraction_operation": "sub_node"
    }
)
graph.add_edge("add_node", "router2")
graph.add_edge("sub_node", "router2")
graph.add_conditional_edges(
    "router2",
    decision_node2,
    {
        # Edge: Node format
        "addition_operation2": "add_node2",
        "subtraction_operation2": "sub_node2"
    }
)

graph.add_edge("add_node2", END)
graph.add_edge("sub_node2", END)

app = graph.compile()

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
initial_state = AgentState(number1 = 10,number2=5, operation1="-",operation2="+", number3=7, number4=8)
result = app.invoke(initial_state)
print(result["result"])
print(result["result2"])