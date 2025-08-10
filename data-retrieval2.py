from typing import TypedDict, NotRequired
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os
import pandas as pd
import psycopg2

# Load environment variables
load_dotenv()
Deep_seek_R1_key = os.getenv("Deep_seek_R1_key")

# ---- Agent State ----
class AgentState(TypedDict):
    system_prompt: SystemMessage
    db_schema: HumanMessage
    user_query: HumanMessage
    transformation_request: NotRequired[HumanMessage]
    instructions: NotRequired[str]
    schema_message: NotRequired[HumanMessage]
    query_message: NotRequired[HumanMessage]
    response: NotRequired[AIMessage]
    db_name: NotRequired[str]
    query_results: NotRequired[pd.DataFrame]

# ---- LLM model ----
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Deep_seek_R1_key,
    model="deepseek/deepseek-r1:free",
)

# ---- Schema Extraction Node ----
def schema_node(state: AgentState) -> AgentState:
    """Fetch DB schema from PostgreSQL automatically."""
    pg_host = os.getenv("PG_HOST")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_db   = os.getenv("PG_DB")
    pg_user = os.getenv("PG_USER")
    pg_pass = os.getenv("PG_PASS")

    try:
        conn = psycopg2.connect(
            host=pg_host, port=pg_port,
            database=pg_db, user=pg_user, password=pg_pass
        )

        schema_sql = """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name;
        """
        df = pd.read_sql_query(schema_sql, conn)
        conn.close()

        schema_str = df.to_string(index=False)
        state["db_schema"] = HumanMessage(content=schema_str)

    except Exception as e:
        state["db_schema"] = HumanMessage(content=f"Error fetching schema: {str(e)}")

    return state

# ---- Context Node ----
def context_node(state: AgentState) -> AgentState:
    state["system_prompt"] = SystemMessage(
        content="You are an expert SQL generator. Given a DB schema and a natural language question, return only the SQL query without explanation."
    )
    schema_text = state["db_schema"].content
    state["instructions"] = 'use Table name in this format schema_name."table_name" to avoid ambiguity.'
    state["schema_message"] = HumanMessage(content=f"Database schema:\n{schema_text}")
    return state

# ---- Query Generation Node ----
def query_node(state: AgentState) -> AgentState:
    query_text = state["user_query"].content
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

# ---- Query Execution Node ----
def executor_node(state: AgentState) -> AgentState:
    sql_query = state["response"].content.strip()
    pg_host = os.getenv("PG_HOST")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_db   = os.getenv("PG_DB")
    pg_user = os.getenv("PG_USER")
    pg_pass = os.getenv("PG_PASS")

    try:
        conn = psycopg2.connect(
            host=pg_host, port=pg_port,
            database=pg_db, user=pg_user, password=pg_pass
        )
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        state["query_results"] = df

    except Exception as e:
        state["query_results"] = pd.DataFrame({"error": [str(e)]})

    return state

# ---- Transformation Node ----
def transform_node(state: AgentState) -> AgentState:
    """Apply transformations to query_results based on transformation_request."""

    state["transformation_request"] = input("Any transformation to apply on the results? ")
    df = state.get("query_results")
    if not isinstance(df, pd.DataFrame):
        return state

    if "transformation_request" not in state or not state["transformation_request"].content.strip():
        return state

    instruction = state["transformation_request"].content.lower()

    try:
        if "uppercase" in instruction:
            df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        if "lowercase" in instruction:
            df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        if "divide" in instruction:
            num = int(instruction.split("divide by")[-1].strip())
            df = df.applymap(lambda x: x / num if isinstance(x, (int, float)) else x)
        if "date format" in instruction:
            fmt = "%Y-%m-%d"
            df = df.applymap(lambda x: x.strftime(fmt) if hasattr(x, "strftime") else x)

        state["query_results"] = df

    except Exception as e:
        state["query_results"] = pd.DataFrame({"error": [str(e)]})

    return state

# ---- Build Graph ----
graph = StateGraph(AgentState)
graph.add_node("schema", schema_node)
graph.add_node("context", context_node)
graph.add_node("query", query_node)
graph.add_node("execute", executor_node)
graph.add_node("transform", transform_node)

graph.add_edge(START, "schema")
graph.add_edge("schema", "context")
graph.add_edge("context", "query")
graph.add_edge("query", "execute")
graph.add_edge("execute", "transform")
graph.add_edge("transform", END)

compiled_graph = graph.compile()

# ---- Run in terminal ----
if __name__ == "__main__":
    db_name = os.getenv("PG_DB")
    while True:
        user_question = input("\nEnter your natural language query (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        state = {
            "db_name": db_name,
            "user_query": HumanMessage(content=user_question),
        }
        result = compiled_graph.invoke(state)

        print("\nGenerated SQL Query:\n")
        print(result["response"].content)

        print("\nFinal Data:\n")
        print(result["query_results"])
