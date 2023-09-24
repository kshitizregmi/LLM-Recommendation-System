from llm import Palm2API
import chromadb


model = Palm2API()

# Initialize the ChromaDB client
client = chromadb.PersistentClient('chromadb')

# Specify the name of the existing collection you want to read
collection_name = "vector_db"

# Get the collection
collection = client.get_collection(collection_name)

# Get user preference and encode it to a vector
user_preference = input("What kind of News do you like? ")
_, vector = model.encode_text_to_embedding_batched(sentences=[user_preference])

# Query ChromaDB for similar articles based on user preference
n_results = 5
results = collection.query(
    query_embeddings=vector.tolist(),
    n_results=n_results
)

# Display the query results
print(results)
