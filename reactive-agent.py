from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os


'''This code works if the llm selected supports tools.'''

load_dotenv()
Deep_seek_R1_key = os.getenv("Deep_seek_R1_key")


class AgentState(TypedDict):
    """Define the state structure for the agent."""
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int) -> int:
    """Add two numbers."""
    return a + b

#List of tools available to the agent
tools = [add]

# Initialize the LLM with the tools
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Deep_seek_R1_key,
    model="deepseek/deepseek-r1:free",
).bind_tools(tools)


# Define the state graph
def model_call(state:AgentState) -> AgentState:
    """Function to call the model with the current state."""
    system_prompt = SystemMessage(
        content="You are a helpful assistant that can perform basic arithmetic operations.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state:AgentState) -> str:
    """Decision node to check if the agent should continue."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "model_call"
    else:
        return "tool_node"
    return END
tool_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools",tool_node)
graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent",should_continue,{"continue":"tools","end":END},)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def  print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if  isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","add 3,5")]}
print_stream(app.stream(inputs,stream_mode = "values"))