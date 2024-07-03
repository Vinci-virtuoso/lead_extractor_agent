# LEAD EXTRACTOR AGENT 
This is a simple lead extractor agentic workflow that allows user do the following:
Upload the input.txt file containing a list of email domains.
Automatically scrapes google and saves the result to a csv file.
Upload the csv file which automatically index the csv file into the postgresql database.
Use AI Agent to query the database in natural language to find the Head of the company and write the name back to the input.txt file


## Project Structure

- `main.py`: Main application file which runs the fastapi edndpoints
- `extractor_agent.py`: Agent for extraction tasks
- `init.sql`: SQL initialization script for the database
- `Dockerfile`: Instructions for building the Docker image
- `docker-compose.yaml`: Docker Compose configuration
- `.dockerignore`: Specifies files to ignore in Docker context
- `.gitignore`: Specifies files to ignore in Git
- `.env.example`: Example environment variable file
- `input.txt`: Input data file
- `google-engine/`: Folder containing scrapped leads from serper api engine
- `requirements.txt`: Python dependencies

## Prerequisites

- Docker
- Docker Compose


## Setup

1. Clone the repository:

2. Replace.env.example with .env and add all appropriate KEYS

3. Build and run the containers: 'Docker-compose up --build' 

4. Run main.py to access the FastAPI server at http://localhost:8000


