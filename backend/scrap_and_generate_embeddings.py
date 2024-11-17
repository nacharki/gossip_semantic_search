# External imports
import json
import logging

# Internal imports
from utils_scrap import get_rss_feed_content, generate_embeddings_articles
from gemini_embeddings import initialize_embeddings_and_chromadb

# Configure logger to display log messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Initialize RSS feed URLs from public.fr and vsd.fr
    rss_public_urls = [
        "https://www.public.fr/feed",
        "https://www.public.fr/people/feed",
        "https://www.public.fr/tele/feed",
        "https://www.public.fr/mode/feed",
        "https://www.public.fr/people/familles-royales/feed",
    ]

    rss_vsd_urls = [
        "https://vsd.fr/actu-people/feed/",
        "https://vsd.fr/tele/feed/",
        "https://vsd.fr/societe/feed/",
        "https://vsd.fr/culture/feed/",
        "https://vsd.fr/loisirs/feed/",
    ]

    # Extract content from RSS feeds
    articles = []
    for url in rss_public_urls + rss_vsd_urls:
        content = get_rss_feed_content(url)
        if content:
            articles.extend(content)
            logger.info("Successfully extracted content from %s", url)
        else:
            logger.warning("Failed to extract content from %s", url)

    # Save all content to a JSON file
    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

    # Initialize the GeminiEmbeddingFunction and ChromaDB collection
    embedding_function, chroma_db = initialize_embeddings_and_chromadb(document_mode=True)

    # Generate embeddings for articles and store them in ChromaDB
    generate_embeddings_articles(articles, embedding_function, chroma_db)

    logger.info("Succesfully scrapped and generated embeddings for articles.")
