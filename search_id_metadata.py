# Import the necessary libraries
import chromadb

# Initialize the ChromaDB client
client = chromadb.PersistentClient('chromadb')

# Specify the name of the existing collection you want to read
collection_name = "vector_db"

# Get the collection
collection = client.get_collection(collection_name)

# List of IDs you want to search for metadata
ids_to_search = ["01111a48-3502-5021-8096-bc9293797d54"]  # Replace with your desired IDs

# Iterate through the list of IDs and retrieve metadata for each ID
for id_to_search in ids_to_search:
    metadata = collection.get(ids=id_to_search)
    if metadata:
        print(f"Metadata for ID '{id_to_search}':")
        print(metadata)
    else:
        print(f"Metadata for ID '{id_to_search}' not found.")

