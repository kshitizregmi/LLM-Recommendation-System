from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llm import Palm2API

# Initialize the ChromaDB client
client = chromadb.PersistentClient('chromadb')

# Specify the name of the existing collection you want to read
collection_name = "vector_db"

# Get the collection
collection = client.get_collection(collection_name)

# Initialize the FastAPI app
app = FastAPI()

class DataSchema(BaseModel):
    """
    Data schema for the request payload.
    """
    article_id: str

def find_metadata_text(id_to_search: str) -> str:
    """
    Retrieve metadata text for the specified article ID.

    Args:
        id_to_search (str): The article ID to search for.

    Returns:
        str: The concatenated metadata text.

    Raises:
        HTTPException: If metadata is not found or 'metadatas' field is missing.
    """
    # Retrieve metadata for the specified ID
    metadata = collection.get(id_to_search)

    if metadata:
        print(f"Metadata for ID '{id_to_search}':")

        # Extract text information from the 'metadatas' field
        docs = metadata.get('metadatas')
        if docs:
            text = "\n".join([f"{doc['headline'].lower()}\n{doc['abstract'].lower()}\n{doc['lead_paragraph'].lower()}" for doc in docs])
            return text
        else:
            raise HTTPException(status_code=400, detail="No 'metadatas' found in the metadata.")
    else:
        raise HTTPException(status_code=404, detail=f"Metadata for ID '{id_to_search}' not found.")

model = Palm2API()

@app.post("/recommend")
def recommend(item: DataSchema) -> dict:
    """
    Endpoint to recommend articles based on the provided article ID.

    Args:
        item (DataSchema): Request payload containing the article ID.

    Returns:
        dict: Recommended articles and metadata.

    Raises:
        HTTPException: If an error occurs during the recommendation process.
    """
    try:
        aid = [item.article_id]
        doc = find_metadata_text(aid)
        _, vector = model.encode_text_to_embedding_batched(sentences=[doc])
        n_results = 5
        results = collection.query(
            query_embeddings=vector.tolist(),
            n_results=n_results
        )

        response = {
            'to_recommend': aid,
            'recommended_ids': results['ids'],
            'metadatas': results['metadatas'],
            'distances': results['distances'],
        }

        # Display the query results
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
