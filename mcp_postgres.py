# mcp_postgres_server.py
from mcp.server import Server
from mcp.types import ToolResponse
import psycopg2
import pandas as pd

server = Server("postgres-server")

@server.tool("run_sql", description="Run a SQL query on a PostgreSQL database")
def run_sql(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    query: str
) -> ToolResponse:
    """
    Executes a SQL query on a given PostgreSQL database and returns the result as CSV text.
    """
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        df = pd.read_sql_query(query, conn)
        conn.close()

        return ToolResponse(content=df.to_csv(index=False))
    except Exception as e:
        return ToolResponse(content=f"ERROR: {str(e)}")

if __name__ == "__main__":
    # Runs MCP server over WebSocket (default localhost:8765)
    server.run()
