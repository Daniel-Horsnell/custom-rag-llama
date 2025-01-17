### Install 
Install Ollama and cuda first

`%%capture --no-stderr`
`%pip install --quiet -U langchain langchain_community tiktoken langchain-nomic "nomic[local]" langchain-ollama scikit-learn langgraph tavily-python bs4`

`pip install langchain-nomic`

`ollama pull llama3.2:3b-instruct-fp16`

### Using
Add docuents to the data_folder, change the file extenstions in the vector_store.add_documents_from_folder() to match yoour data, and run the rag_implemmentation.py. Once it has run once you can comment out the line in the rag_implementation.py file vector_store.add_documents_from_folder(). This only needs to be done once unless your data changes and will speed up the program.

