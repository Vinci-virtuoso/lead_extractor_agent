import os
import math
import json
import requests
from pydantic import BaseModel, ValidationError, validator
import dotenv

config = dotenv.dotenv_values(".env")
SERPER_API_KEY = config['SERPER_API_KEY']

class SearchResult(BaseModel):
    title: str
    snippet: str

    @validator('snippet')
    def must_contain_keywords(cls, value, field):
        keywords = ["CEO", "Chief Executive Officer", "Executive Director", "Managing Partner", "President"]
        if not any(keyword in value for keyword in keywords):
            raise ValueError(f"{field.name} must contain one of the following: {keywords}")
        return value

def get_ceo_info(email_url):
    search_query = f"CEO of {email_url}"
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": search_query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return []

    response_json = response.json()
    organic_results = response_json.get('organic', [])
    return organic_results

def validate_and_filter_results(data):
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

input_file = 'input.txt'
output_directory = 'google_engine' 

with open(input_file, 'r') as file:
    lines = file.readlines()

total_urls = len(lines)
urls_per_file = math.ceil(total_urls / 10)

for i, line in enumerate(lines):
    at_pos = line.find('@')
    if at_pos != -1:
        email_url = line[at_pos+1:].strip()
        api_response_data = get_ceo_info(email_url)
        validated_results = validate_and_filter_results(api_response_data)
        
        file_number = i // urls_per_file + 1
        display_results_to_file(validated_results, email_url, output_directory, file_number)
