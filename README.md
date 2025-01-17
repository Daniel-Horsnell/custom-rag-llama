### Install 
Install Ollama and cuda first

`%%capture --no-stderr
%pip install --quiet -U langchain langchain_community tiktoken langchain-nomic "nomic[local]" langchain-ollama scikit-learn langgraph tavily-python bs4`

`pip install langchain-nomic`

`ollama pull llama3.2:3b-instruct-fp16`

