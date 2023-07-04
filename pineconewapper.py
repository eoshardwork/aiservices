import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)

index = pinecone.Index("aiservices")


class PineconeVectorstore:
    def __init__(self):
        pass

    def from_documents(self, docs, embeddings, index_name, namespace):
        vector_db = Pinecone.from_documents(
            docs, embeddings, index_name="aiservices", namespace=namespace
        )
        return vector_db

    def add_documents(self, docs, namespace):
        return Pinecone.add_documents(
            docs, namespace=namespace
        )

    def from_existing_index(self, embeddings, namespeace):
        return Pinecone.from_existing_index(
            "aiservices", embedding=embeddings, namespace=namespeace
        )

    def delete_all_vectors(self, namespace):
        index.delete(delete_all=True, namespace=namespace)
