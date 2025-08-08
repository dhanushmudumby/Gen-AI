from typing import Dict, TypedDict,List,Union
from langgraph.graph import StateGraph,START, END
import os
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain.schema import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma


load_dotenv()


API_KEY = os.getenv("API_KEY")


class AgentState(TypedDict):
    messages:List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model = "deepseek/deepseek-r1:free"
    )

PERSIST_DIR = "./conversation_memory_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)



def process_text(state:AgentState) -> AgentState:
    """Function to process the text and generate a response."""
    last_message = state["messages"][-1]
    doc = Document(page_content=last_message.content,
                   metadata={"role": type(last_message).__name__})

    # Store only the new message
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    vector_db.add_documents(chunks)

    retrieved_docs = vector_db.similarity_search(last_message.content, k=3)
    past_context = "\n".join([d.page_content for d in retrieved_docs])

    # Append retrieved context as part of the system prompt
    augmented_messages = [
        HumanMessage(content=f"Past relevant conversation:\n{past_context}"),
        *state["messages"]
    ]

    response = llm.invoke(augmented_messages)
    state["messages"].append(AIMessage(content=response.content))

    print(f"\nAI Response: {response.content}")
    print("current sate",state["messages"])
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process_text", process_text)
graph.add_edge(START, "process_text")
graph.add_edge("process_text", END)
agent = graph.compile()


user_input = input("Enter your message: " )
convo_history = []

while user_input!="exit":
    convo_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": convo_history})
    convo_history = result["messages"]
    user_input = input("Enter your message: " )

