import os
import math
import json
import requests
import dotenv
import logging
import uvicorn
from enum import Enum
from typing import List
from langchain import hub
from pydantic import BaseModel
from langserve import add_routes
from langchain.schema import Document
from fastapi import FastAPI, HTTPException
from store_factory import get_vector_store
from dotenv import find_dotenv, load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.indexes import SQLRecordManager, index
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ValidationError, validator
from langchain.tools.retriever import create_retriever_tool
from fastapi import FastAPI, HTTPException,UploadFile, File
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import Tool
from langchain_experimental.tools import PythonREPLTool

load_dotenv(find_dotenv())
config = dotenv.dotenv_values(".env")
OPENAI_API_KEY = config['OPENAI_API_KEY']
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class SearchResult(BaseModel):
    title: str
    snippet: str

    @validator('snippet')
    def must_contain_keywords(cls, value, field):
        keywords = ["CEO", "Chief Executive Officer", "Executive Director", "Managing Partner", "President"]
        if not any(keyword in value for keyword in keywords):
            raise ValueError(f"{field.name} must contain one of the following: {keywords}")
        return value

def get_ceo_info(email_url: str) -> List[dict]:
    search_query = f"CEO of {email_url}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        return []
    return response.json().get('organic', [])

def validate_and_filter_results(data: List[dict]) -> List[SearchResult]:
    valid_results = []
    for item in data:
        try:
            valid_result = SearchResult(**item)
            valid_results.append(valid_result)
        except ValidationError:
            continue
    return valid_results
def display_results_to_file(results, email_url, output_directory, file_number):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_file_path = os.path.join(output_directory, f"data_part_{file_number}.txt")
    
    with open(output_file_path, "a", encoding='utf-8') as text_file:
        text_file.write(f"Email URL: {email_url}\n")
        for result in results:
            text_file.write(f"Title: {result.title}\nSnippet: {result.snippet}\n")
        text_file.write("\n" + "-"*150 + "\n")

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value
try:
    USE_ASYNC = os.getenv("USE_ASYNC", "False").lower() == "true"
    if USE_ASYNC:
        print("Async project used")

    POSTGRES_DB = get_env_variable("POSTGRES_DB")
    POSTGRES_USER = get_env_variable("POSTGRES_USER")
    POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD")
    DB_HOST = get_env_variable("DB_HOST")
    DB_PORT = get_env_variable("DB_PORT")
    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")

    CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

    embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)

    mode = "async" if USE_ASYNC else "sync"
    pgvector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name="testcollection",
        mode=mode,
    )
    namespace = "testcollection"
    record_manager = SQLRecordManager(namespace, db_url=CONNECTION_STRING)
    record_manager.create_schema()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class DocumentRequest(BaseModel):
        page_content: str
        metadata: dict

    class CleanupMethod(str, Enum):
        incremental = "incremental"
        full = "full"

except Exception as e:
    print(f"An error occurred during initialization: {e}")
    
retriever = pgvector_store.as_retriever()
model=ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0)
prompt = hub.pull("hwchase17/react")
retriever_tool=create_retriever_tool(
    retriever,
    "company-search",
    description="Use this tool when retrieving information about a company C.E.O or C.E.O equivalent")

python_repl = PythonREPLTool()
repl_tool = Tool(
    name="python_repl",
    description=" Executes Python code for file reading and writing operations, allowing for manipulation of the file system.",
    func=python_repl.run,
)
tools = [retriever_tool,repl_tool]
agent = create_react_agent(model, tools, prompt)
agentExecutor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, handle_parsing_errors=True, max_iterations=4
)
class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/index")
async def index_documents(docs_request: list[DocumentRequest], cleanup: CleanupMethod = CleanupMethod.incremental) -> dict:
    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in docs_request
    ]

    result = index(
        documents,
        record_manager,
        pgvector_store,
        cleanup=cleanup.value,
        source_id_key="source",
    )
    return result

@app.get("/search-ceo/", response_model=List[SearchResult])
async def search_ceo(email_url: str):
    api_response_data = get_ceo_info(email_url)
    if not api_response_data:
        raise HTTPException(status_code=400, detail="Failed to fetch data or no data available for this query.")
    validated_results = validate_and_filter_results(api_response_data)
    return validated_results

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    lines = content.decode("utf-8").splitlines()
    total_urls = len(lines)
    urls_per_file = math.ceil(total_urls / 10)
    output_directory = 'google_engine'

    for i, line in enumerate(lines):
        at_pos = line.find('@')
        if at_pos != -1:
            email_url = line[at_pos+1:].strip()
            api_response_data = get_ceo_info(email_url)
            validated_results = validate_and_filter_results(api_response_data)
            
            file_number = i // urls_per_file + 1
            display_results_to_file(validated_results, email_url, output_directory, file_number)

    return {"message": "File processed successfully"}
add_routes(
    app, agentExecutor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ), path="/agent"
)

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)