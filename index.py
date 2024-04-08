import requests
import dotenv
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,TextLoader
config = dotenv.dotenv_values(".env")
load_dotenv(find_dotenv())
OPENAI_API_KEY= config['OPENAI_API_KEY']
embedding = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)

loader = DirectoryLoader('./FAQ', glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1840,
)

documents = text_splitter.split_documents(docs)
docs_data = [doc.dict() for doc in documents]

url = "http://localhost:8000/index?cleanup=full"
response = requests.post(url, json=docs_data)
print(response.json())
