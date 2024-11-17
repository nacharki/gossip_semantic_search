# External imports
import logging
import argparse

# Internal imports
from utils_search import search_similar_articles, display_results
from gemini_embeddings import initialize_embeddings_and_chromadb

# Configure logger to display log messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Set up argument parser to read query and number of results
    parser = argparse.ArgumentParser(description="Search similar articles")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("n_results", type=int, help="Number of results to return")

    # Parse arguments
    args = parser.parse_args()

    query = args.query
    n_results = args.n_results

    # Initialize the GeminiEmbeddingFunction and ChromaDB collection with query mode
    embedding_function, chroma_db = initialize_embeddings_and_chromadb(document_mode=False)

    # Search for articles
    agg_metadata = search_similar_articles(query, embedding_function, chroma_db, n_results=n_results)

    # Display results
    display_results(agg_metadata)
