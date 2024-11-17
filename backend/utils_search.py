# External imports
from typing import List, Dict
from collections import defaultdict
from chromadb import Collection

# Internal imports
from gemini_embeddings import GeminiEmbeddingFunction


def is_duplicate_article(existing: dict, new_metadata: dict) -> bool:
    """Check if article is an exact duplicate based on metadata

    parameters:
    ------------
    existing : dict
        The existing metadata of the article.
    new_metadata : dict
        The new metadata of the article to compare.

    returns
    ------------
    bool: True if the articles are duplicates, False otherwise."""
    fields = ["title", "creator", "pub_date", "description"]
    return all(existing[field].strip() == new_metadata.get(field, "").strip() for field in fields)


def aggregate_results(results: dict) -> List[Dict]:
    """Aggregate the search results to combine metadata and content snippets for each article

    Parameters:
    ------------
    results : dict
        The search results returned by the ChromaDB query function.

    Return
    ------------
    aggregated_results : List[Dict]
        A list of dictionaries containing aggregated metadata for the search results.
    """

    # Create defaultdict for aggregation, categories and urls are sets to avoid duplicates
    aggregated = defaultdict(
        lambda: {
            "title": "",
            "creator": "",
            "categories": set(),
            "description": "",
            "pub_date": "",
            "content_snippets": [],
            "similarity_scores": [],
            "urls": set(),
        }
    )

    # Get result lists from ChromaDB query
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    # Zip and iterate over metadata, document, and distance
    for metadata, document, distance in zip(metadatas, documents, distances):
        # Create compound key from title and creator
        title = metadata.get("title", "").strip()
        creator = metadata.get("creator", "").strip()
        key = f"{title}::{creator}"  # Compound key

        if not title:
            # Skip articles without titles
            continue

        # Skip if exact duplicate already exists
        if key in aggregated and is_duplicate_article(aggregated[key], metadata):
            continue

        # Update metadata for new or non-duplicate article
        aggregated[key].update(
            {
                "title": title,
                "creator": creator,
                "description": metadata.get("description", ""),
                "pub_date": metadata.get("pub_date", ""),
                "categories": metadata.get("categories", ""),
            }
        )

        # Add content and score
        aggregated[key]["content_snippets"].append(document)
        aggregated[key]["similarity_scores"].append(distance)

    # Convert to list and calculate averages
    aggregated_results = []
    for article in aggregated.values():
        # Calculate average similarity
        if article["similarity_scores"]:
            article["avg_similarity"] = sum(article["similarity_scores"]) / len(article["similarity_scores"])
            # Remove raw scores
            del article["similarity_scores"]
        aggregated_results.append(article)

    # Sort by average similarity (highest first)
    aggregated_results.sort(key=lambda x: x.get("avg_similarity", 0), reverse=True)

    return aggregated_results


def search_similar_articles(
    query: str,
    embedding_function: GeminiEmbeddingFunction,
    chroma_db: Collection,
    n_results: int = 10,
) -> List[Dict]:
    """Search and retrieve similar articles based on the user query for a given number of results

    Parameters:
    ------------
    query : str
        The user query to search for similar articles, e.g., "famille royale", "people", "TPMP" etc.
    embedding_function : GeminiEmbeddingFunction
        The embedding function to generate embeddings for the query with document_mode=False.
    chroma_db : Collection
        The ChromaDB collection to query for similar articles.
    n_results : int
        The number of results to retrieve (default is 10).

    Returns:
    ------------
    agg_metadata : List[Dict]
        A list of dictionaries containing aggregated metadata for the search results.
    """
    # Generate and process embedding for the query
    embedding = embedding_function.get_embeddings(query)
    embedding = embedding_function.process_embedding(embedding)

    # Query ChromaDB for similar articles
    results = chroma_db.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    # Aggregate the results
    agg_metadata = aggregate_results(results)

    return agg_metadata


def display_results(articles: List[Dict]):
    """Display search results in a formatted way"""
    for i, article in enumerate(articles):
        print(f"Result {i+1}")
        print(f"Title: {article['title']}")
        print(f"Author: {article['creator']}")
        print(f"Publication Date: {article['pub_date']}")
        print(f"Categories: {article['categories']}")
        print(f"Description: {article['description']}")
        print(f"Average Similarity Score: {article['avg_similarity']:.4f}")
        print("Content Snippets:")
        for snippet in article["content_snippets"]:
            print(f"- {snippet[:200]}...")
        print("\n" + "-" * 50 + "\n")
