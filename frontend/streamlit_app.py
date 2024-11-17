# External imports
import os
import sys
import streamlit as st

# Set the working directory to the backend
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.abspath(os.path.join(current_dir, "..", "backend"))
sys.path.append(backend_path)

# Internal imports
from utils_search import search_similar_articles  # noqa: E402
from gemini_embeddings import Initialize_embeddings_and_chromadb  # noqa: E402


def get_search_results(query: str, n_results: int):
    """Direct search function that uses backend functionality"""
    try:
        # Initialize the GeminiEmbeddingFunction and ChromaDB collection with query mode
        embedding_function, chroma_db = Initialize_embeddings_and_chromadb(document_mode=False)
        # Search for articles and aggregate metadata
        agg_metadata = search_similar_articles(query, embedding_function, chroma_db, n_results)

        return agg_metadata
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def display_results(aggregated_results):
    for article in aggregated_results:
        st.markdown(f"### {article['title']}")
        st.markdown(f"**Author:** {article['creator']}")
        st.markdown(f"**Publication Date:** {article['pub_date']}")
        st.markdown(f"**Categories:** {article['categories']}")
        st.markdown(f"**Description:** {article['description']}")
        st.markdown(f"**Average Similarity Score:** {article['avg_similarity']:.4f}")
        if "url" in article:
            st.markdown(f"[Read Full Article]({article['url']})")

        with st.expander("Show Content Snippets"):
            for snippet in article["content_snippets"]:
                st.write(snippet)
        st.write("---")


if __name__ == "__main__":
    st.set_page_config(page_title="Gossip Semantic Search", layout="wide")
    st.title("📰 Gossip Semantic Search")
    st.write("Search for the latest articles from VSD and Public.")

    query = st.text_input("Enter your search query:", value="")
    n_results = st.slider("Number of results to display:", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        if query.strip():
            with st.spinner("Searching..."):
                results = get_search_results(query, n_results)
                if results:
                    display_results(results)
                else:
                    st.warning("No results found for your query.")
        else:
            st.error("Please enter a valid query.")
