import os
import logging
from typing import List, Tuple
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from google.api_core import retry
import chromadb
from chromadb import Collection, Documents, EmbeddingFunction, Embeddings


# Load environment variables and GOOGLE_API_KEY
load_dotenv()

# Configure logger to display log messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    GeminiEmbeddingFunction is a custom embedding function that generates embeddings for documents or queries
    using the Google Generative AI model. It can be used in document retrieval or query retrieval tasks.

    Attributes:
    ------------
    api_key : str
        The API key for accessing the Gemini API services, to be used for generating embeddings.
    document_mode : bool
        A flag indicating whether to generate embeddings for documents (True) or queries (False).

    Methods:
    ------------
    __init__(document_mode: bool)
        Initializes the GeminiEmbeddingFunction with the specified document mode.

    __call__(input: Documents) -> Embeddings
        Generates embeddings for the given input documents or queries.
    """

    def __init__(self, document_mode: bool = True):
        # Set the document_mode flag
        self.document_mode = document_mode
        # Check if the API key is available
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key is required to use the GeminiEmbeddingFunction.")
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        # Initialize the embedding model for document or query retrieval
        self.model = "models/embedding-001"

    def get_embeddings(self, input: Documents) -> Embeddings:
        """get embeddings for the given input documents or queries."""
        # Configure the retry policy for handling transient errors
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        # Generate embeddings for the input documents or queries
        response = genai.embed_content(
            model=self.model,
            content=input,
            task_type="retrieval_document" if self.document_mode else "retrieval_query",
            request_options=retry_policy,
        )

        return response["embedding"]

    @staticmethod
    def process_embedding(embedding: Embeddings) -> List[float]:
        """Process and validate embedding format"""
        if isinstance(embedding, list) and len(embedding) == 1:
            # If the embedding is a list with a single element, extract the embedding
            embedding = embedding[0]
        elif isinstance(embedding, np.ndarray):
            # If the embedding is a numpy array, convert it to a list
            embedding = embedding.tolist()

        return embedding


def initialize_embeddings_and_chromadb(
    document_mode: bool = True,
) -> Tuple[GeminiEmbeddingFunction, Collection]:
    """Initialize the GeminiEmbeddingFunction and ChromaDB collection for storing embeddings and metadata.

    Parameters:
    ------------
        document_mode : bool
            Whether to generate embeddings in document mode if True, or query mode if False (default is True).

    Returns
    ------------
        Tuple[GeminiEmbeddingFunction, chromadb.Collection]
            A tuple containing the GeminiEmbeddingFunction and ChromaDB collection
    """

    # Initialize the GeminiEmbeddingFunction to generate embeddings on document retrieval mode
    embedding_function = GeminiEmbeddingFunction(document_mode=document_mode)

    # Initialize ChromaDB client and create a collection in ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chromadb")
    chroma_db = chroma_client.get_or_create_collection(name="articles", embedding_function=embedding_function)

    logger.info("Initialized ChromaDB client and created a chroma database (collection) in ChromaDB.")

    return embedding_function, chroma_db
