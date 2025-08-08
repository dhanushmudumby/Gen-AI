import langchain
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
print("testing the branch protection")


