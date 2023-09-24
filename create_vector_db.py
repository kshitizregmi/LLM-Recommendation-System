import pandas as pd
import chromadb
from llm import Palm2API

# Load the "Vertex AI Embeddings for Text" model
from vertexai.preview.language_models import TextEmbeddingModel

# Load the dataset
df = pd.read_csv('nyt-metadata.csv', nrows=5000, na_values=['[]', "", " "])

# Define the required columns
required_columns = ['uri', 'headline', 'abstract', 'lead_paragraph']
df = df[required_columns]
df.dropna(inplace=True)

# Extract article IDs from URIs
df["article_id"] = df['uri'].str.split("/").apply(lambda x: x[-1])

# Extract the main part of the headline
df.headline = df.headline.apply(lambda x: eval(x)['main'])

# Create a corpus from lowercase text
corpus = df['headline'].str.lower() + "\n" + df['abstract'].str.lower() + "\n" + df['lead_paragraph'].str.lower()

# Encode the corpus using batching
model = Palm2API()
is_successful, embeddings = model.encode_text_to_embedding_batched(sentences=corpus)

# Create a list of article IDs and metadata dictionaries
article_ids = df.article_id.astype(str).tolist()
metadata = df.to_dict(orient='records')

# Initialize the ChromaDB client and collection
client = chromadb.PersistentClient('chromadb')
collection = client.create_collection("vector_db")

# Add the embeddings, metadata, and IDs to the ChromaDB collection
collection.add(
    embeddings=embeddings.tolist(),
    metadatas=metadata,
    ids=article_ids
)