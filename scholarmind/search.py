"""
ScholarMind — Search Engine
Supports semantic search, hybrid search, and filtered queries using Endee.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from endee import Endee

import config


class ScholarSearch:
    """Multi-mode search engine backed by Endee vector database."""

    def __init__(self):
        """Initialize search engine with Endee client and embedding model."""
        # ── Endee Client ──
        self.client = Endee(config.ENDEE_AUTH_TOKEN) if config.ENDEE_AUTH_TOKEN else Endee()
        self.client.set_base_url(config.ENDEE_BASE_URL)

        # ── Embedding Model ──
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        # ── TF-IDF Vectorizer (for hybrid search) ──
        self.vectorizer = self._load_vectorizer()

        # ── Endee Indexes ──
        self.semantic_index = self.client.get_index(name=config.PAPERS_INDEX_NAME)
        self.hybrid_index = self.client.get_index(name=config.PAPERS_HYBRID_INDEX_NAME)

    def _load_vectorizer(self) -> TfidfVectorizer | None:
        """Load saved TF-IDF vectorizer."""
        path = os.path.join(os.path.dirname(__file__), "sample_data", "tfidf_vectorizer.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _embed_query(self, query: str) -> list[float]:
        """Generate dense embedding for a query string."""
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def _sparse_query(self, query: str) -> tuple[list[int], list[float]]:
        """Generate sparse TF-IDF vector for a query string."""
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded. Run ingest.py first.")
        sparse_vec = self.vectorizer.transform([query])
        indices = sparse_vec.indices.tolist()
        values = sparse_vec.data.tolist()
        return indices, values

    # ── Search Methods ──────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = None,
        ef: int = None,
    ) -> list[dict]:
        """
        Pure semantic similarity search.
        Embeds the query and finds closest papers by cosine similarity.
        """
        top_k = top_k or config.DEFAULT_TOP_K
        ef = ef or config.DEFAULT_EF
        
        query_vector = self._embed_query(query)
        results = self.semantic_index.query(
            vector=query_vector,
            top_k=top_k,
            ef=ef,
            include_vectors=False,
        )
        return self._format_results(results)

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        ef: int = None,
    ) -> list[dict]:
        """
        Hybrid search combining dense embeddings + sparse TF-IDF vectors.
        Captures both semantic meaning and exact keyword relevance.
        """
        top_k = top_k or config.DEFAULT_TOP_K
        ef = ef or config.DEFAULT_EF
        
        query_vector = self._embed_query(query)
        sparse_indices, sparse_values = self._sparse_query(query)
        
        results = self.hybrid_index.query(
            vector=query_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=top_k,
            ef=ef,
            include_vectors=False,
        )
        return self._format_results(results)

    def filtered_search(
        self,
        query: str,
        category: str | None = None,
        area: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        top_k: int = None,
        ef: int = None,
    ) -> list[dict]:
        """
        Semantic search with Endee filter operators.
        Supports:
          - $eq for exact category/area match
          - $range for year filtering (normalized to 0-999)
        """
        top_k = top_k or config.DEFAULT_TOP_K
        ef = ef or config.DEFAULT_EF
        
        query_vector = self._embed_query(query)
        
        # Build filter list
        filters = []
        if category:
            filters.append({"category": {"$eq": category}})
        if area:
            filters.append({"area": {"$eq": area}})
        if year_min is not None or year_max is not None:
            min_norm = self._normalize_year(year_min or 2010)
            max_norm = self._normalize_year(year_max or 2025)
            filters.append({"year": {"$range": [min_norm, max_norm]}})
        
        results = self.semantic_index.query(
            vector=query_vector,
            top_k=top_k,
            ef=ef,
            filter=filters if filters else None,
            include_vectors=False,
        )
        return self._format_results(results)

    def multi_area_search(
        self,
        query: str,
        areas: list[str],
        top_k: int = None,
    ) -> list[dict]:
        """
        Search within multiple areas using $in filter operator.
        Example: areas=["nlp", "cv"] to search across NLP and CV papers.
        """
        top_k = top_k or config.DEFAULT_TOP_K
        
        query_vector = self._embed_query(query)
        
        filters = [{"area": {"$in": areas}}]
        
        results = self.semantic_index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filters,
            include_vectors=False,
        )
        return self._format_results(results)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_year(year: int, min_year: int = 2010, max_year: int = 2025) -> int:
        """Normalize year to 0-999 range for Endee $range filter."""
        normalized = int(((year - min_year) / (max_year - min_year)) * 999)
        return max(0, min(999, normalized))

    @staticmethod
    def _format_results(results: list) -> list[dict]:
        """Format Endee query results into clean dicts."""
        formatted = []
        for item in results:
            formatted.append({
                "id": item.get("id", ""),
                "similarity": round(item.get("similarity", 0.0), 4),
                "title": item.get("meta", {}).get("title", "Unknown"),
                "authors": item.get("meta", {}).get("authors", "Unknown"),
                "abstract": item.get("meta", {}).get("abstract", ""),
                "year": item.get("meta", {}).get("year", ""),
                "category": item.get("meta", {}).get("category", ""),
                "keywords": item.get("meta", {}).get("keywords", ""),
            })
        return formatted

    def get_index_stats(self) -> dict:
        """Get statistics for both indexes."""
        stats = {}
        for name in [config.PAPERS_INDEX_NAME, config.PAPERS_HYBRID_INDEX_NAME]:
            try:
                index = self.client.get_index(name=name)
                info = index.describe()
                stats[name] = info
            except Exception as e:
                stats[name] = {"error": str(e)}
        return stats


# ── CLI Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]ScholarMind[/bold cyan] — Search Engine Demo",
        border_style="bright_blue",
    ))

    engine = ScholarSearch()

    # Demo 1: Semantic Search
    console.print("\n[bold yellow]═══ Semantic Search ═══[/bold yellow]")
    query = "How do transformers process sequences without recurrence?"
    console.print(f"Query: [italic]{query}[/italic]\n")
    results = engine.semantic_search(query, top_k=5)
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("Title", width=50)
    table.add_column("Year", width=6)
    table.add_column("Similarity", width=10)
    for i, r in enumerate(results, 1):
        table.add_row(str(i), r["title"], str(r["year"]), f"{r['similarity']:.4f}")
    console.print(table)

    # Demo 2: Filtered Search
    console.print("\n[bold yellow]═══ Filtered Search (CV papers, 2020+) ═══[/bold yellow]")
    query2 = "image segmentation methods"
    results2 = engine.filtered_search(query2, category="computer_vision", year_min=2020, top_k=5)
    
    table2 = Table(show_header=True, header_style="bold")
    table2.add_column("#", width=3)
    table2.add_column("Title", width=50)
    table2.add_column("Year", width=6)
    table2.add_column("Similarity", width=10)
    for i, r in enumerate(results2, 1):
        table2.add_row(str(i), r["title"], str(r["year"]), f"{r['similarity']:.4f}")
    console.print(table2)

    # Demo 3: Multi-Area Search
    console.print("\n[bold yellow]═══ Multi-Area Search (NLP + CV) ═══[/bold yellow]")
    query3 = "attention mechanism in neural networks"
    results3 = engine.multi_area_search(query3, areas=["nlp", "cv", "transformers"], top_k=5)
    
    table3 = Table(show_header=True, header_style="bold")
    table3.add_column("#", width=3)
    table3.add_column("Title", width=50)
    table3.add_column("Area", width=15)
    table3.add_column("Similarity", width=10)
    for i, r in enumerate(results3, 1):
        table3.add_row(str(i), r["title"], r["category"], f"{r['similarity']:.4f}")
    console.print(table3)

    # Demo 4: Hybrid Search
    console.print("\n[bold yellow]═══ Hybrid Search ═══[/bold yellow]")
    query4 = "reinforcement learning Atari game playing DQN"
    console.print(f"Query: [italic]{query4}[/italic]\n")
    results4 = engine.hybrid_search(query4, top_k=5)
    
    table4 = Table(show_header=True, header_style="bold")
    table4.add_column("#", width=3)
    table4.add_column("Title", width=50)
    table4.add_column("Year", width=6)
    table4.add_column("Similarity", width=10)
    for i, r in enumerate(results4, 1):
        table4.add_row(str(i), r["title"], str(r["year"]), f"{r['similarity']:.4f}")
    console.print(table4)
