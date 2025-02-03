# Gossip Semantic Search

A semantic search engine for French gossip websites **vsd.fr** and **public.fr**. This application allows users to search for articles semantically, providing more accurate and context-aware search results compared to traditional keyword-based search engines.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-green.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/chromadb-0.4+-orange.svg)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/gemini--api-1.0+-purple.svg)](https://ai.google.dev/)
[![BeautifulSoup](https://img.shields.io/badge/beautifulsoup4-4.0+-yellow.svg)](https://www.crummy.com/software/BeautifulSoup/)

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Primary development language
- **Google Gemini API**: For text embeddings generation
- **ChromaDB**: Vector database for efficient similarity search
- **BeautifulSoup4**: Web scraping and HTML parsing
- **Pandas**: Data manipulation and processing
- **NumPy**: Numerical computations
- **Requests**: HTTP requests handling
- **Streamlit**: Interactive web interface


## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Semantic search using embeddings](#semantic-search-using-embeddings)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [License](#license)



## Features

- **Semantic Search**: Use gemini embedding models to understand meanings of text embeddings.
- **Scrapping and data extraction**: Extracts articles from RSS feeds of **vsd.fr** and **public.fr** using beautifulsoup4.
- **Vector Database**: Utilizes ChromaDB to store and retrieve embeddings efficiently.
- **Interactive Frontend**: Provides a user-friendly interface built with Streamlit for easy user interaction.
- **Metadata Display**: Presents search results along with relevant metadata like title, author, publication date, and snippets.



## Project Structure

```
├── frontend/                  
    ├── streatmlit_app.py                       # Streamlit app for the frontend 
├── backend/                                
    ├── gemini_embedding.py                     # Module for gemini embedding functions
    ├── scrap_and_generate_embeddings.py        # Python script for scrapping articles and generate embeddings
    ├── search_similarity.py                    # Python script for searching similarity given a query
    ├── utils_scrap.py                          # Utility functions for scrapping
    ├── utils_search.py                         # Utility functions for searching similarities
    ├── tests_backend/                          # Tests for some functions in the backend
        ├── test_scrap.py
        ├── test_search.py
├── requirements.txt                            # Python dependencies
├── black.sh                                    # black script for formatting
├── pylint.sh                                   # pylint script for code quality, score: 8.40/10
├── flake8.shr                                  # flake8 script for checking errors
├── Dockerfile                                  # Dockerfile to create docker image - not needed ow
├── README.md                                   # Project documentation

```



## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/gossip-semantic-search.git
   cd gossip-semantic-search
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Semantic search using embeddings

1. **Run the scrappign and embedding generation script**

   The `scrap_and_generate_embeddings.py` script fetches articles from the RSS feeds and processes them.

   ```bash
   python scrap_and_generate_embeddings.py
   ```

   This will:

   - Extract articles from the specified RSS feeds.
   - Save the articles to `articles.json`.
   - Generate embeddings using the `GeminiEmbeddingFunction` using gemini//model-001 and process them..
   - Store embeddings and metadata in ChromaDB.

2. **Run the similarity search for query**

   The `search_similarity.py` script will search for a input 'query' and number of results to be shown 'n_result' 

   ```bash
   python search_similarity.py 'famille royale' 20
   ```

   This will:

   - Input the text query.
   - Generate embeddings for the input query with `GeminiEmbeddingFunction`  and process them.
   - Run brut-force search in chromaDB and retrieve similar documents based on similarity score.
   - Aggregate metadata for more comprehensive display of results

### Running the Streamlit App

1. **Start the Streamlit Application**

   ```bash
   streamlit run frontend/streamlit_app.py
   ```

2. **Using the App**

   - Open the provided URL in  `http://localhost:8501`.
   - Enter a search query in the input field and the maximal number of results to show
   - Click the **Search** button to retrieve results.
   - View the articles along with their metadata and content snippets.


## Examples of use

First example: searching for 'TPMP' program and displaying results:

![Alt text](images\tpmp_search.png)

Second example: search for 'famille royale' and displaying results:

![Alt text](images\famille_royale_search.png)


## License

This project is licensed under the MIT License.


## Future Improvements

- **Expand Data Sources:**

  - Include more gossip or news websites to enrich the dataset.
  - Use more sophisticated searching methods such as scaNN or LSH for large documents datasets.

- **Improve the User Interface:**

  - Add more interactive elements and better styling to the Streamlit app.

- **Implement Advanced Search Features:**

  - Allow filtering by date, author, or categories.

- **Miscellaneous coding improvements:**
  - Handle some edge-cases and exceptions in the code.
  - Optimize embeddings articles with token chunking.
  - Include more exhaustive tests: integration and end-to-end tests.

---
