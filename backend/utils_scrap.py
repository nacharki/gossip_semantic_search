# External imports
import logging
import requests
from bs4 import BeautifulSoup
from chromadb import Collection

# Internal imports
from gemini_embeddings import GeminiEmbeddingFunction

# Configure logger to display log messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_rss_feed_content(url: str) -> list:
    """
    Extract content from an RSS feed URL and returns a list of dictionaries containing the extracted content such
    as title, author, categories, description, publication date, and main content of the article. If an error occurs
    during the extraction process, it logs the error and returns None.

    parameters:
    ------------
    url: str
        URL of the RSS feed

    returns:
    ------------
    list
        List of dictionaries containing the extracted content
            title: str - Title of the article
            creator: str - Author of the article
            categories: list - List of categories the article belongs to
            description: str - Short description of the article
            pub_date: str - Publication date of the article
            content: str - Main content of the article
    """
    # Set user-agent header to avoid blocking by the server
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"  # noqa: E501
    }

    # Make a request to the RSS feed URL, get the resoponse, and parse the XML content
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response.encoding = "utf-8"
    except requests.exceptions.HTTPError as errh:
        logging.error("HTTP Error: %s for URL: %s", errh, url)
        return None
    except requests.exceptions.RequestException as err:
        logging.error("Error: %s for URL: %s", err, url)
        return None

    # Initialize BeautifulSoup object to parse the XML content
    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")
    content_list = []

    for item in items:
        try:
            # Extract relevant metadata from the RSS feed item
            title = item.find("title").get_text(strip=True) if item.find("title") else "N/A"
            creator = item.find("dc:creator").get_text(strip=True) if item.find("dc:creator") else "N/A"
            categories = [category.get_text(strip=True) for category in item.find_all("category")]
            description = item.find("description").get_text(strip=True) if item.find("description") else "N/A"
            publication_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else "N/A"
            content_encoded = (
                item.find("content:encoded").get_text(strip=True) if item.find("content:encoded") else "N/A"
            )

            # Clean up HTML content
            content_text = ""
            if content_encoded != "N/A":
                # Parse content:encoded field if available
                content_encoded_soup = BeautifulSoup(content_encoded, "html.parser")
                for script_or_style in content_encoded_soup(["script", "style"]):
                    script_or_style.decompose()
                content_text = content_encoded_soup.get_text(separator=" ", strip=True).replace("\n", " ")
            else:
                # Fallback to description
                description_soup = BeautifulSoup(description, "html.parser")
                for script_or_style in description_soup(["script", "style"]):
                    script_or_style.decompose()
                content_text = description_soup.get_text(separator=" ", strip=True).replace("\n", " ")

            content_list.append(
                {
                    "title": title,
                    "creator": creator,
                    "categories": categories,
                    "description": description,
                    "pub_date": publication_date,
                    "content": content_text,
                }
            )
        except Exception as e:
            logging.error("Error parsing item %s", e)
            continue

    return content_list


def chunk_text(text, max_chunk_size: int = 9000) -> list:
    """Split text into chunks smaller than max_chunk_size bytes
    NOTE: Token-Based Chunking can be used for more accurate chunking than byte-based chunking

    Parameters:
    ------------
    text : str
        The text to be split into chunks

    max_chunk_size : int
        The maximum size of each chunk in bytes (default is 9000)

    Returns
    ------------
    list
        A list of text chunks
    """
    # Encode text to bytes and check if it fits in a single chunk
    text_bytes = text.encode("utf-8")
    if len(text_bytes) <= max_chunk_size:
        return [text]

    # Else, split the text into chunks based on the max_chunk_size
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_size = 0

    # Iterate over sentences and create chunks
    for sentence in sentences:
        sentence_size = len((sentence + ". ").encode("utf-8"))
        if current_size + sentence_size > max_chunk_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


def generate_embeddings_articles(
    articles: list, embedding_function: GeminiEmbeddingFunction, chroma_db: Collection
) -> None:
    """Generate embeddings for a list of articles and store them in ChromaDB.
    - NOTE: To be parallelized for large datasets

    Parameters:
    ------------
    articles : list
        A list of dictionaries containing articles with metadata and content
    embedding_function : GeminiEmbeddingFunction
        The embedding function to generate embeddings for the articles
    chroma_db : Collection
        The ChromaDB collection to store the embeddings and metadata
    """
    # Generate embeddings and store them in ChromaDB
    for idx, article in enumerate(articles):
        title = article["title"]
        content = article["content"]

        # Split content into chunks
        content_chunks = chunk_text(content)

        # Process each chunk
        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_id = f"{idx}-{chunk_idx}"
            # Generate and process embeddings for the chunk
            embedding = embedding_function.get_embeddings(chunk)
            embedding = embedding_function.process_embedding(embedding)

            # Store chunk with metadata
            chroma_db.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[
                    {
                        "title": title,
                        "creator": article["creator"],
                        "categories": (", ".join(article["categories"]) if article["categories"] else ""),
                        "description": article["description"],
                        "pub_date": article["pub_date"],
                        "chunk_index": chunk_idx,
                        "total_chunks": len(content_chunks),
                    }
                ],
                embeddings=[embedding],
            )

    logger.info("Embeddings have been successfully stored in ChromaDB.")

    return None
