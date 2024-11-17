import unittest
from unittest.mock import MagicMock
from backend.utils_search import search_similar_articles, aggregate_results


class TestUtilsSearch(unittest.TestCase):

    def setUp(self):
        self.query = "test query"
        self.embedding_function = MagicMock()
        self.chroma_db = MagicMock()
        self.n_results = 5

        # Mock embedding function
        self.embedding_function.get_embeddings.return_value = "mock_embedding"
        self.embedding_function.process_embedding.return_value = "processed_embedding"

        # Mock ChromaDB query results
        self.mock_results = {
            "metadatas": [
                [
                    {
                        "title": "Article 1",
                        "creator": "Author 1",
                        "description": "Desc 1",
                        "pub_date": "2023-01-01",
                        "categories": "Category 1",
                    },
                    {
                        "title": "Article 2",
                        "creator": "Author 2",
                        "description": "Desc 2",
                        "pub_date": "2023-01-02",
                        "categories": "Category 2",
                    },
                ]
            ],
            "documents": [["Content snippet 1", "Content snippet 2"]],
            "distances": [[0.1, 0.2]],
        }
        self.chroma_db.query.return_value = self.mock_results

    def test_search_similar_articles(self):
        # Call the function
        results = search_similar_articles(self.query, self.embedding_function, self.chroma_db, self.n_results)

        # Assertions
        self.embedding_function.get_embeddings.assert_called_once_with(self.query)
        self.embedding_function.process_embedding.assert_called_once_with("mock_embedding")
        self.chroma_db.query.assert_called_once_with(
            query_embeddings=["processed_embedding"],
            n_results=self.n_results,
            include=["metadatas", "documents", "distances"],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Article 1")
        self.assertEqual(results[1]["title"], "Article 2")

    def test_aggregate_results(self):
        # Call the function
        aggregated_results = aggregate_results(self.mock_results)

        # Assertions
        self.assertEqual(len(aggregated_results), 2)
        self.assertEqual(aggregated_results[0]["title"], "Article 1")
        self.assertEqual(aggregated_results[0]["avg_similarity"], 0.1)
        self.assertEqual(aggregated_results[1]["title"], "Article 2")
        self.assertEqual(aggregated_results[1]["avg_similarity"], 0.2)


if __name__ == "__main__":
    unittest.main()
