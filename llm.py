from typing import List, Optional, Generator, Tuple
import numpy as np
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import math
import env

# Load the "Vertex AI Embeddings for Text" model
from vertexai.preview.language_models import TextEmbeddingModel

class Palm2API:
    """
    A class for working with the Vertex AI "Vertex AI Embeddings for Text" model.
    """
    def __init__(self):
        """
        Initialize the Palm2API by loading the TextEmbeddingModel.
        """
        # Initialize the TextEmbeddingModel
        self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

    def generate_batches(self, sentences: List[str], batch_size: int) -> Generator[List[str], None, None]:
        """
        Generate batches of sentences.

        Args:
            sentences (List[str]): List of sentences to be batched.
            batch_size (int): Batch size.

        Yields:
            Generator[List[str], None, None]: A generator yielding batches of sentences.
        """
        for i in range(0, len(sentences), batch_size):
            yield sentences[i: i + batch_size]

    def encode_text_to_embedding_batched(self, sentences: List[str], api_calls_per_second: int = 10, batch_size: int = 5) -> Tuple[List[bool], np.ndarray]:
        """
        Encode text to embeddings using batching and multithreading.

        Args:
            sentences (List[str]): List of sentences to encode.
            api_calls_per_second (int): Number of API calls per second.
            batch_size (int): Batch size.

        Returns:
            Tuple[List[bool], np.ndarray]: A tuple containing a list of boolean values indicating success for each sentence
                and a NumPy array of embeddings for successful sentences.
        """
        embeddings_list: List[List[float]] = []

        # Prepare the batches using a generator
        batches = self.generate_batches(sentences, batch_size)

        seconds_per_job = 1 / api_calls_per_second

        with ThreadPoolExecutor() as executor:
            futures = []
            for batch in tqdm(batches, total=math.ceil(len(sentences) / batch_size), position=0):
                futures.append(executor.submit(functools.partial(self.encode_texts_to_embeddings), batch))
                time.sleep(seconds_per_job)

            for future in futures:
                embeddings_list.extend(future.result())

        is_successful = [embedding is not None for embedding in embeddings_list]
        embeddings_list_successful = np.squeeze(np.stack([embedding for embedding in embeddings_list if embedding is not None]))
        return is_successful, embeddings_list_successful

    def encode_texts_to_embeddings(self, sentences: List[str]) -> List[Optional[List[float]]]:
        """
        Encode a list of sentences to embeddings using the model.

        Args:
            sentences (List[str]): List of sentences to encode.

        Returns:
            List[Optional[List[float]]]: A list of embeddings for each sentence. If encoding fails for a sentence, the
                corresponding entry is None.
        """
        try:
            embeddings = self.model.get_embeddings(sentences)
            return [embedding.values for embedding in embeddings]
        except Exception:
            return [None for _ in range(len(sentences))]
