import sys

from langchain_ollama import ChatOllama

from vector_store import VectorStoreManager
from langchain_core.messages import HumanMessage, SystemMessage

# Format docuents
def format_docs(docs):
    return "\n\n".join(doc[0].page_content for doc in docs)

# Prompt
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

folder_path="./data_folder"

if __name__ == "__main__":
    folder_path = './data_folder'
    # Create vector store with persistant path
    vector_manager = VectorStoreManager('./vectorDB')
    
    # Add documents from folder
    vector_manager.add_documents_from_folder(folder_path)

    # Make persistant
    vector_manager.save_vector_store()

    # Create ll
    local_llm = "llama3.2:3b-instruct-fp16"
    llm = ChatOllama(model=local_llm, temperature=0.8)

    # Retrieve information
    try:
        while True:
            # Wait for the user input
            user_input = input("Enter something (or press Ctrl+C to exit): ")

            # Run the provided function
            docs = vector_manager.retrieve(user_input)
            docs_txt = format_docs(docs)

            # If no docuents
            if(len(docs) == 0):
                rag_prompt_formatted = rag_prompt.format(context="No relevant information in database. Please mmake the user aware of this.", question=user_input)
            # If documents
            else:
                rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=user_input)
            
            # Send prommpt to ll and print
            generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
            print(generation.content)



    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C
        print("\nExiting program.")
        sys.exit(0)
