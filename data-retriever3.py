from typing import TypedDict, NotRequired
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from mcp.client import Client
import os
import pandas as pd

# Load env vars
load_dotenv()
Deep_seek_R1_key = os.getenv("Deep_seek_R1_key")

# Load DB schema from CSV
schema_path = "C:/Users/dmudu/OneDrive/Desktop/AI-Impl/Gen-AI/db-schema.csv"
db_schema = pd.read_csv(schema_path).to_string(index=False)

# ---- Agent State ----
class AgentState(TypedDict):
    system_prompt: SystemMessage
    db_schema: HumanMessage
    user_query: HumanMessage
    instructions: NotRequired[str]
    schema_message: NotRequired[HumanMessage]
    query_message: NotRequired[HumanMessage]
    response: NotRequired[AIMessage]
    db_name: NotRequired[str]
    query_results: NotRequired[str]

# ---- LLM model ----
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Deep_seek_R1_key,
    model="deepseek/deepseek-r1:free",
)

# ---- Nodes ----
def context_node(state: AgentState) -> AgentState:
    state["system_prompt"] = SystemMessage(
        content="You are an expert SQL generator. Given a DB schema and a natural language question, return only the SQL query without explanation."
    )
    schema_text = state["db_schema"].content if isinstance(state["db_schema"], HumanMessage) else state["db_schema"]
    state["db_schema"] = HumanMessage(content=schema_text)
    state["instructions"] = 'Use table name format schema_name."table_name" to avoid ambiguity.'
    state["schema_message"] = HumanMessage(content=f"Database schema:\n{schema_text}")
    return state

def query_node(state: AgentState) -> AgentState:
    query_text = state["user_query"].content if isinstance(state["user_query"], HumanMessage) else state["user_query"]
    state["query_message"] = HumanMessage(content=f"User question:\n{query_text}")
    messages = [
        state["system_prompt"],
        HumanMessage(content=state["instructions"]),
        state["schema_message"],
        state["query_message"]
    ]
    llm_response = model.invoke(messages)
    state["response"] = llm_response
    return state

def mcp_postgres_node(state: AgentState) -> AgentState:
    sql_query = state["response"].content.strip()

    # DB credentials from .env
    pg_host = os.getenv("PG_HOST")
    pg_port = int(os.getenv("PG_PORT", "5432"))
    pg_db   = os.getenv("PG_DB")
    pg_user = os.getenv("PG_USER")
    pg_pass = os.getenv("PG_PASS")

    # Connect to MCP server
    mcp_client = Client("ws://localhost:8765")  # Your MCP server address
    result = mcp_client.call_tool(
        "run_sql",
        host=pg_host,
        port=pg_port,
        database=pg_db,
        user=pg_user,
        password=pg_pass,
        query=sql_query
    )

    state["query_results"] = result["content"]
    return state

# ---- Graph ----
graph = StateGraph(AgentState)
graph.add_node("context", context_node)
graph.add_node("query", query_node)
graph.add_node("db_exec", mcp_postgres_node)

graph.add_edge(START, "context")
graph.add_edge("context", "query")
graph.add_edge("query", "db_exec")
graph.add_edge("db_exec", END)

compiled_graph = graph.compile()

# ---- CLI Loop ----
if __name__ == "__main__":
    db_name = input("Enter the database name: ")
    while True:
        user_question = input("\nEnter your natural language query (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        state = {
            "db_name": db_name,
            "db_schema": HumanMessage(content=db_schema),
            "user_query": HumanMessage(content=user_question)
        }
        result = compiled_graph.invoke(state)

        print("\nGenerated SQL Query:\n")
        print(result["response"].content)
        print("\nQuery Execution Result:\n")
        print(result["query_results"])
