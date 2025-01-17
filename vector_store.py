from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.schema import Document
from langchain_nomic.embeddings import NomicEmbeddings
import os
import glob

class VectorStoreManager:
    def __init__(self, path, chunk_size=1000, chunk_overlap=200, relevance_threshold=0.7, embedding_model="nomic-embed-text-v1.5"):
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._vectorstore = SKLearnVectorStore(
            embedding=NomicEmbeddings(model=embedding_model, inference_mode="local"),
            persist_path=path
        )
        self._relevance_threshold = relevance_threshold

    def add_documents_from_folder(self, folder_path, file_extensions=(".md", ".mdx")):
        """Add documents from a folder to the vector store. Update extensions to use other formats."""

        # Get all files in folder
        file_paths = []
        for ext in file_extensions:
            file_paths.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))

        # Read docs
        docs = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                docs.append(Document(page_content=content, metadata={"source": file_path}))

        # Split documents and add to vector store
        doc_splits = self._text_splitter.split_documents(docs)
        self._vectorstore.add_documents(doc_splits)

    def retrieve(self, query, k=4):
        docs_and_relevance = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
        # Filter document out if it is below relevence threshold
        return [item for item in docs_and_relevance if item[1] >= self._relevance_threshold]
        
    def save_vector_store(self):
        self._vectorstore.persist()
