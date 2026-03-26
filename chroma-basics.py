# https://realpython.com/chromadb-vector-database/

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)

documents = [
    "The latest iPhone model comes with impressive features and a powerful camera.",
    "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
    "Einstein's theory of relativity revolutionized our understanding of space and time.",
    "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
    "The American Revolution had a profound impact on the birth of the United States as a nation.",
    "Regular exercise and a balanced diet are essential for maintaining good physical health.",
    "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
    "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
    "Startup companies often face challenges in securing funding and scaling their operations.",
    "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
]

genres = [
    "technology",
    "travel",
    "science",
    "food",
    "history",
    "fitness",
    "art",
    "climate change",
    "business",
    "music",
]

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],
    metadatas=[{"genre": g} for g in genres]
)


def main():
    print("\nChroma Basics\n")

    # Single Query
    query_results = collection.query(
        query_texts=["Find me some delicious food!"],
        n_results=1
    )

    print(f"Query Keys: {query_results.keys()}\n")
    print(f"Query IDs: {query_results["ids"]}\n")
    print(f"Query Distances: {query_results["distances"]}\n")
    print(f"Query Metadatas: {query_results["metadatas"]}\n")
    print(f"Query Docs: {query_results["documents"]}\n")
    
    
    # Querying with multiple questions
    question = ["Teach me about history",
             "What's going on in the world?"]

    query_results = collection.query(
        query_texts=question,
        include=["documents", "distances"],
        n_results=2
    )

    for i in range(len(query_results["documents"])):
        print(f"Query:{i} - {question[i]}\n")
        print(f"Query Results: {query_results["documents"][i]}\n")


    # Querying with filters
    print(
        collection.query(
        query_texts=["Teach me about music history"],
        where={"genre": {"$eq": "music"}},
        n_results=1,
        )
    )

    print("Deleting Collection ...")
    client.delete_collection(name=COLLECTION_NAME)
    client.clear_system_cache()
    

if __name__ == "__main__":
    main()
