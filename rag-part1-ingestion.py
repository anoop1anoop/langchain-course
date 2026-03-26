import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()


def main():
    print("Starting Ingestion for RAG Model  .... \n")
    loader = TextLoader("rag-part1-mediumblog1.txt")
    document = loader.load()

    print("Splitting Doc...\n")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    print("Storing in Vector Store...\n")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Finished Ingestion for RAG Model!\n")


if __name__ == "__main__":
    main()
