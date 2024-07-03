import os
import re
from typing import TypedDict, Optional, Type
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph import StateGraph, END
from langchain.callbacks.manager import CallbackManagerForToolRun

load_dotenv(find_dotenv())
class CompanyLeadExtractor:

    def __init__(self):
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.DB_PORT = os.getenv('DB_PORT')
        self.POSTGRES_HOST = os.getenv('POSTGRES_HOST')
        self.POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE')
        self.CONNECTION_STRING = f"postgresql://{self.POSTGRES_USERNAME}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.DB_PORT}/{self.POSTGRES_DATABASE}"
        self.COLLECTION_NAME = os.getenv('COLLECTION_NAME')
        
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.vectorstore = PGVector(
            collection_name=self.COLLECTION_NAME,
            connection=self.CONNECTION_STRING,
            embeddings=self.embeddings,
            use_jsonb=True
        )
        self.retriever = self.vectorstore.as_retriever()
        
        self.model = ChatGroq(
            temperature=0,
            groq_api_key=self.GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        
        self.template = """
        Analyze the provided context to extract the Head the company using these rules which you are to strictly adhere to:
        
        1. If a CEO is mentioned, extract the CEO's name.
        2. If no CEO is mentioned, look for roles such as Owner, President, Vice-President, Founder, or similar, and extract the name with the highest weight.
        
        Weights:
        - CEO = 1
        - President = 0.9
        - Founder = 0.8
        - Owner = 0.7
        - Vice-President = 0.6
        
        Context: {context}
        Question: {question}
        
        You will get a $100 tip if you provide the correct answer, reply only with the final name from your analysis.
        """
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        
        self.retrieval_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def extract_ceo_name(self, query: str) -> str:
        return self.retrieval_chain.invoke(query)

class FileInfoInput(BaseModel):
    domain: str = Field(description="Domain to search in the file")
    ceo_name: str = Field(description="CEO's name to append")

class FileManipulatorTool(BaseTool):
    name = "file_manipulator"
    description = "Appends CEO's name to the line just above the matching domain in input.txt"
    args_schema: Type[BaseModel] = FileInfoInput

    def _run(self, domain: str, ceo_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        file_path = 'input.txt'
        if not os.path.exists(file_path):
            return "File not found"

        new_content = []
        found = False
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if domain in lines[i]:
                    new_content.append(ceo_name + "\n") 
                    found = True
                new_content.append(lines[i])

        if found:
            with open(file_path, 'w') as file:
                file.writelines(new_content)
            return "CEO's name appended successfully"
        else:
            return "Domain not found"

    async def _arun(self, domain: str, ceo_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._run(domain, ceo_name, run_manager)

class LeadWorkflow:

    class AgentState(TypedDict):
        query: str
        ceo_name: str
        domain_url: str
        generation: str

    def __init__(self, extractor: CompanyLeadExtractor, manipulator: FileManipulatorTool):
        self.extractor = extractor
        self.manipulator = manipulator
        self.workflow = StateGraph(self.AgentState)

        self.workflow.add_node("entry", self.entry)
        self.workflow.add_node("extract_ceo_name", self.extract_ceo_name)
        self.workflow.add_node("file_writer", self.file_writer)
        self.workflow.add_edge("entry", "extract_ceo_name")
        self.workflow.add_edge("extract_ceo_name", "file_writer")
        self.workflow.add_edge("file_writer", END)
        self.workflow.set_entry_point("entry")
        self.graph = self.workflow.compile()

    def entry(self, query: str) -> str:
        return query

    def parse_domain_url(self, query: str) -> Optional[str]:
        match = re.search(r"([\w.]+)", query, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_ceo_name(self, state: AgentState) -> AgentState:
        query = state["query"]
        ceo_name = self.extractor.extract_ceo_name(query)
        domain_url = self.parse_domain_url(query)
        if domain_url is None:
            raise ValueError("Could not extract domain from query.")
        state["ceo_name"] = ceo_name
        state["domain_url"] = domain_url
        return state

    def file_writer(self, state: AgentState) -> AgentState:
        ceo_name = state["ceo_name"]
        domain_url = state["domain_url"]
        response = self.manipulator.invoke({"domain": domain_url, "ceo_name": ceo_name})
        state["generation"] = response
        return state

    def process_query(self, query: str) -> AgentState:
        return self.graph.invoke({"query": query})

if __name__ == "__main__":
    extractor = CompanyLeadExtractor()
    manipulator = FileManipulatorTool()
    workflow = LeadWorkflow(extractor, manipulator)

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        result = workflow.process_query(query)
        print(result["generation"])
