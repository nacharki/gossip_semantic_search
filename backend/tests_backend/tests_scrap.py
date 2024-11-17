# External imports
import os
import sys
import unittest
from unittest.mock import MagicMock
from chromadb import Collection

# Set the working directory to the backend
sys.path.extend([os.path.abspath("."), os.path.abspath("./backend")])

# Internal imports
from backend.utils_scrap import (  # noqa: E402
    get_rss_feed_content,
    chunk_text,
    generate_embeddings_articles,
)
from gemini_embeddings import GeminiEmbeddingFunction  # noqa: E402


class TestScrapping(unittest.TestCase):
    """
    This class tests the functions in the utils_scrap.py file.
    """

    def test_get_rss_feed_content(self):
        """test the get_rss_feed_content function to ensure it returns a list of items for a valid RSS feed URL"""
        # Test with a valid RSS feed URL
        url = "https://www.public.fr/feed"
        content = get_rss_feed_content(url)
        self.assertIsInstance(content, list)
        self.assertGreater(len(content), 0)
        # Test with an invalid URL
        url = "https://invalid-url.com/feed"
        content = get_rss_feed_content(url)
        self.assertIsNone(content)

    def test_chunk_text(self):
        """test the chunk_text function to ensure it splits a text into chunks"""
        text = "Test example: this sentence for testing should split this text into chunks."
        chunks = chunk_text(text, max_chunk_size=20)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))


class TestGenerateEmbeddingsArticles(unittest.TestCase):
    def setUp(self):
        self.articles = [
            {
                "title": "Test Article 1",
                "creator": "Author 1",
                "categories": ["Category 1", "Category 2"],
                "description": "This is a test description for article 1.",
                "pub_date": "2023-01-01",
                "content": "This is the main content of the test article 1. It has multiple sentences. This is another sentence.",  # noqa: E501
            },
            {
                "title": "Test Article 2",
                "creator": "Author 2",
                "categories": ["Category 3"],
                "description": "This is a test description for article 2.",
                "pub_date": "2023-01-02",
                "content": "This is the main content of the test article 2. It has multiple sentences. This is another sentence.",  # noqa: E501
            },
        ]
        self.embedding_function = MagicMock(spec=GeminiEmbeddingFunction)
        self.embedding_function.get_embeddings.return_value = [0.1, 0.2, 0.3]
        self.embedding_function.process_embedding.return_value = [0.1, 0.2, 0.3]
        self.chroma_db = MagicMock(spec=Collection)

    def test_generate_embeddings_articles(self):
        generate_embeddings_articles(self.articles, self.embedding_function, self.chroma_db)

        # Check if embeddings were generated and stored in ChromaDB for two articles
        self.assertEqual(self.embedding_function.get_embeddings.call_count, 2)
        self.assertEqual(self.embedding_function.process_embedding.call_count, 2)
        self.assertEqual(self.chroma_db.add.call_count, 2)

        # # Check if the correct data was passed to ChromaDB
        # expected_calls = [
        #     {
        #         "ids": ["0"],
        #         "documents": ["This is the main content of the test article 1."],
        #         "metadatas": [
        #             {
        #                 "title": "Test Article 1",
        #                 "creator": "Author 1",
        #                 "categories": "Category 1, Category 2",
        #                 "description": "This is a test description for article 1.",
        #                 "pub_date": "2023-01-01",
        #                 "chunk_index": 0,
        #                 "total_chunks": 2,
        #             }
        #         ],
        #         "embeddings": [[0.1, 0.2, 0.3]],
        #     },
        #     {
        #         "ids": ["1"],
        #         "documents": ["It has multiple sentences. This is another sentence."],
        #         "metadatas": [
        #             {
        #                 "title": "Test Article 1",
        #                 "creator": "Author 1",
        #                 "categories": "Category 1, Category 2",
        #                 "description": "This is a test description for article 1.",
        #                 "pub_date": "2023-01-01",
        #                 "chunk_index": 1,
        #                 "total_chunks": 2,
        #             }
        #         ],
        #         "embeddings": [[0.1, 0.2, 0.3]],
        #     },
        #     {
        #         "ids": ["2"],
        #         "documents": ["This is the main content of the test article 2."],
        #         "metadatas": [
        #             {
        #                 "title": "Test Article 2",
        #                 "creator": "Author 2",
        #                 "categories": "Category 3",
        #                 "description": "This is a test description for article 2.",
        #                 "pub_date": "2023-01-02",
        #                 "chunk_index": 0,
        #                 "total_chunks": 2,
        #             }
        #         ],
        #         "embeddings": [[0.1, 0.2, 0.3]],
        #     },
        #     {
        #         "ids": ["13"],
        #         "documents": ["It has multiple sentences. This is another sentence."],
        #         "metadatas": [
        #             {
        #                 "title": "Test Article 2",
        #                 "creator": "Author 2",
        #                 "categories": "Category 3",
        #                 "description": "This is a test description for article 2.",
        #                 "pub_date": "2023-01-02",
        #                 "chunk_index": 1,
        #                 "total_chunks": 2,
        #             }
        #         ],
        #         "embeddings": [[0.1, 0.2, 0.3]],
        #     },
        # ]

        # for call, expected in zip(self.chroma_db.add.call_args_list, expected_calls):
        #     args, kwargs = call
        #     self.assertEqual(kwargs, expected)


if __name__ == "__main__":
    unittest.main()
