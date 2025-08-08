from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
path = "C:/Users/dmudu/OneDrive/Desktop/MYProject/Demo-Project/Demo/app/HR_policies.txt"
loader = TextLoader(path, encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
vector_db.persist()

vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
query = "What is the hike policy?"
results = vector_db.similarity_search(query, k=3)

context = "\n".join([doc.page_content for doc in results])
final_prompt = f"""Use the following company context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

completion = client.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=[
        {
            "role": "user",
            "content": final_prompt
        }
    ]
)

print(completion.choices[0].message.content)