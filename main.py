import json
import pandas as pd
import os
import uvicorn
import logging
import requests
from enum import Enum
from typing import List
from io import StringIO
from pydantic import BaseModel
from collections import defaultdict
from langchain_postgres.vectorstores import PGVector
from langchain.indexes import SQLRecordManager, index
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError, field_validator
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


# Load environment variables
load_dotenv(find_dotenv())

SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('DB_PORT')
POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

class SearchResult(BaseModel):
    title: str
    snippet: str

    @field_validator("snippet")
    def must_contain_keywords(cls, value):
        keywords = ["CEO", "Chief Executive Officer", "Executive Director", "Managing Partner", "President", "Founder"]
        if not any(keyword in value for keyword in keywords):
            raise ValueError("Snippet must contain a relevant keyword.")
        return value

def get_ceo_info(email_url: str) -> List[dict]:
    search_query = f"CEO of {email_url}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        return []
    return response.json().get("organic", [])

def validate_and_filter_results(data: List[dict]) -> List[SearchResult]:
    valid_results = []
    for item in data:
        try:
            valid_result = SearchResult(**item)
            valid_results.append(valid_result)
        except ValidationError:
            continue
    return valid_results

def format_results(all_results):
    # This function aggregates titles and snippets by 'email_url'
    organized_data = defaultdict(list)
    for result in all_results:
        organized_data[result['email_url']].append((result['title'], result['snippet']))
    # Now format the data for output
    formatted_data = []
    for email_url, results in organized_data.items():
        titles = " || ".join(f"{i+1}. {title}" for i, (title, _) in enumerate(results))
        snippets = " || ".join(f"{i+1}. {snippet}" for i, (_, snippet) in enumerate(results))
        formatted_data.append({"Email_URL": email_url, "Title": titles, "Snippet": snippets, "Source": "email_domain_result.csv"})
    return formatted_data

CONNECTION_STRING = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{DB_PORT}/{POSTGRES_DATABASE}"
embeddings =FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    embeddings=embeddings,
    use_jsonb=True
)
namespace = f"pgvector/{COLLECTION_NAME}"

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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/search-ceo/", response_model=List[SearchResult])
async def search_ceo(email_url: str):
    api_response_data = get_ceo_info(email_url)
    if not api_response_data:
        raise HTTPException(status_code=400, detail="Failed to fetch data or no data available for this query.")
    validated_results = validate_and_filter_results(api_response_data)
    return validated_results

@app.post("/upload-list/")
async def upload_file(file: UploadFile = File(...)):
    all_results = []
    with open('input.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            at_pos = line.find('@')
            if at_pos != -1:
                email_url = line[at_pos+1:].strip()
                api_response_data = get_ceo_info(email_url)  # Moved inside the loop
                if api_response_data:
                    validated_results = validate_and_filter_results(api_response_data)
                    for result in validated_results:
                        all_results.append({"email_url": email_url, "title": result.title, "snippet": result.snippet, "Source": "email_domain_result.csv"})

    if not all_results:
        print("No results to write to CSV.")
    else:
        formatted_results = format_results(all_results)
        df = pd.DataFrame(formatted_results)
        output_directory = 'google-engine'
        output_file_path = os.path.join(output_directory, "email_domain_results.csv")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        df.to_csv(output_file_path, index=False)
        print(f"All results saved to {output_file_path}")

    return {"message": "File processed successfully"}

@app.post("/index")
async def index_documents(file: UploadFile = File(...), cleanup: CleanupMethod = CleanupMethod.incremental) -> dict:
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')), encoding='ISO-8859-1')

        # Load the data into documents using DataFrameLoader
        loader = DataFrameLoader(df, page_content_column='Snippet')
        docs = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=612,
            chunk_overlap=103,
        )
        documents = text_splitter.split_documents(docs)

        # Index the documents
        result = index(
            documents,
            record_manager,
            vectorstore,
            cleanup=cleanup.value,
            source_id_key="Source",
        )
        
        return {"status": "success", "indexed_documents": len(documents), "result": result}
    
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)