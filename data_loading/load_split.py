import sys
import os

# Add the project directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from utils.embeddings import PhoBertEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings



import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
    
# Load the documents
def load_documents():
    loader = WebBaseLoader("https://vi.wikipedia.org/wiki/Đại_học_Bách_khoa_Hà_Nội")

    documents = loader.load()
    return documents

# Split the documents into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(documents)
    return chunks

# Storing the vector embeddings in vector database
def store_vector_db(chunks):
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name="vinai/phobert-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    texts = []
    for chunk in chunks:
        if hasattr(chunk, 'text'):
            texts.append(chunk.text)
        else:
            # Handle the case where the chunk does not have a text attribute
            texts.append(str(chunk))
    
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        persist_directory=f"../vector_db",
        collection_name="hust_info",
        metadatas=None
    )

    return vector_db.persist()

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    # print(chunks)
    print("Chunks length: ", len(chunks))
    
    vector_db = store_vector_db(chunks) 
