"""
ScholarMind — Data Ingestion Pipeline
Loads research papers, generates embeddings, and indexes them in Endee.
"""

import json
import time
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from endee import Endee, Precision

import config

console = Console()


def load_papers(path: str) -> list[dict]:
    """Load research papers from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    console.print(f"[green]✓[/green] Loaded {len(papers)} papers from {path}")
    return papers


def generate_dense_embeddings(papers: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Generate dense embeddings from paper titles + abstracts."""
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    
    with console.status("[bold cyan]Generating dense embeddings..."):
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings


def generate_sparse_vectors(papers: list[dict], max_features: int = None) -> tuple:
    """Generate TF-IDF sparse vectors for hybrid search."""
    max_features = max_features or config.SPARSE_DIMENSION
    texts = [f"{p['title']} {p['abstract']} {' '.join(p.get('keywords', []))}" for p in papers]
    
    with console.status("[bold cyan]Generating sparse (TF-IDF) vectors..."):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
    
    console.print(f"[green]✓[/green] Generated TF-IDF vectors with {max_features} features")
    return vectorizer, tfidf_matrix


def normalize_year(year: int, min_year: int = 2010, max_year: int = 2025) -> int:
    """Normalize year to 0-999 range for Endee $range filter."""
    normalized = int(((year - min_year) / (max_year - min_year)) * 999)
    return max(0, min(999, normalized))


def setup_endee_client() -> Endee:
    """Initialize Endee client."""
    client = Endee(config.ENDEE_AUTH_TOKEN) if config.ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(config.ENDEE_BASE_URL)
    console.print(f"[green]✓[/green] Connected to Endee at {config.ENDEE_BASE_URL}")
    return client


def create_indexes(client: Endee):
    """Create Endee indexes for papers."""
    
    # ── Semantic Search Index ──
    try:
        client.delete_index(config.PAPERS_INDEX_NAME)
        console.print(f"[yellow]⟳[/yellow] Deleted existing index: {config.PAPERS_INDEX_NAME}")
    except Exception:
        pass
    
    client.create_index(
        name=config.PAPERS_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        space_type=config.SPACE_TYPE,
        precision=Precision.INT8,
    )
    console.print(f"[green]✓[/green] Created index: [bold]{config.PAPERS_INDEX_NAME}[/bold] (dim={config.EMBEDDING_DIMENSION}, space={config.SPACE_TYPE})")
    
    # ── Hybrid Search Index ──
    try:
        client.delete_index(config.PAPERS_HYBRID_INDEX_NAME)
        console.print(f"[yellow]⟳[/yellow] Deleted existing index: {config.PAPERS_HYBRID_INDEX_NAME}")
    except Exception:
        pass
    
    client.create_index(
        name=config.PAPERS_HYBRID_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        sparse_dim=config.SPARSE_DIMENSION,
        space_type=config.SPACE_TYPE,
    )
    console.print(f"[green]✓[/green] Created hybrid index: [bold]{config.PAPERS_HYBRID_INDEX_NAME}[/bold] (sparse_dim={config.SPARSE_DIMENSION})")


def ingest_semantic_index(client: Endee, papers: list[dict], embeddings: np.ndarray):
    """Upsert paper vectors into the semantic search index."""
    index = client.get_index(name=config.PAPERS_INDEX_NAME)
    
    vectors = []
    for i, paper in enumerate(papers):
        vectors.append({
            "id": paper["id"],
            "vector": embeddings[i].tolist(),
            "meta": {
                "title": paper["title"],
                "authors": ", ".join(paper["authors"][:3]) + ("..." if len(paper["authors"]) > 3 else ""),
                "abstract": paper["abstract"][:300],
                "year": paper["year"],
                "category": paper["category"],
                "keywords": ", ".join(paper.get("keywords", [])),
            },
            "filter": {
                "category": paper["category"],
                "area": paper.get("area", "ml"),
                "year": normalize_year(paper["year"]),
            },
        })
    
    # Batch upsert
    BATCH_SIZE = 20
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
        task = progress.add_task("Upserting to semantic index...", total=len(vectors))
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(batch)
            progress.update(task, advance=len(batch))
    
    console.print(f"[green]✓[/green] Upserted {len(vectors)} vectors to [bold]{config.PAPERS_INDEX_NAME}[/bold]")


def ingest_hybrid_index(client: Endee, papers: list[dict], embeddings: np.ndarray, tfidf_matrix):
    """Upsert paper vectors into the hybrid search index."""
    index = client.get_index(name=config.PAPERS_HYBRID_INDEX_NAME)
    
    vectors = []
    for i, paper in enumerate(papers):
        # Get sparse vector from TF-IDF
        sparse_row = tfidf_matrix[i]
        sparse_indices = sparse_row.indices.tolist()
        sparse_values = sparse_row.data.tolist()
        
        vectors.append({
            "id": paper["id"],
            "vector": embeddings[i].tolist(),
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "meta": {
                "title": paper["title"],
                "authors": ", ".join(paper["authors"][:3]) + ("..." if len(paper["authors"]) > 3 else ""),
                "abstract": paper["abstract"][:300],
                "year": paper["year"],
                "category": paper["category"],
            },
            "filter": {
                "category": paper["category"],
                "area": paper.get("area", "ml"),
                "year": normalize_year(paper["year"]),
            },
        })
    
    # Batch upsert
    BATCH_SIZE = 20
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
        task = progress.add_task("Upserting to hybrid index...", total=len(vectors))
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(batch)
            progress.update(task, advance=len(batch))
    
    console.print(f"[green]✓[/green] Upserted {len(vectors)} vectors to [bold]{config.PAPERS_HYBRID_INDEX_NAME}[/bold]")


def print_index_stats(client: Endee):
    """Print index statistics."""
    table = Table(title="📊 Index Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Index Name", style="cyan")
    table.add_column("Vectors", justify="right", style="green")
    table.add_column("Dimension", justify="right")
    table.add_column("Space", justify="right")
    
    for name in [config.PAPERS_INDEX_NAME, config.PAPERS_HYBRID_INDEX_NAME]:
        try:
            index = client.get_index(name=name)
            info = index.describe()
            table.add_row(
                name,
                str(info.get("vector_count", "N/A")),
                str(info.get("dimension", "N/A")),
                str(info.get("space_type", "N/A")),
            )
        except Exception as e:
            table.add_row(name, f"Error: {e}", "", "")
    
    console.print(table)


def save_vectorizer(vectorizer: TfidfVectorizer):
    """Save TF-IDF vectorizer for later use in search."""
    import pickle
    import os
    path = os.path.join(os.path.dirname(__file__), "sample_data", "tfidf_vectorizer.pkl")
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    console.print(f"[green]✓[/green] Saved TF-IDF vectorizer to {path}")


def main():
    """Main ingestion pipeline."""
    console.print(Panel.fit(
        "[bold cyan]ScholarMind[/bold cyan] — Data Ingestion Pipeline\n"
        "Loading papers → Generating embeddings → Indexing in Endee",
        border_style="bright_blue",
    ))
    
    start_time = time.time()
    
    # Step 1: Load papers
    papers = load_papers(config.SAMPLE_DATA_PATH)
    
    # Step 2: Load embedding model
    console.print("\n[bold]Step 1:[/bold] Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    console.print(f"[green]✓[/green] Loaded model: {config.EMBEDDING_MODEL_NAME}")
    
    # Step 3: Generate embeddings
    console.print("\n[bold]Step 2:[/bold] Generating embeddings...")
    embeddings = generate_dense_embeddings(papers, model)
    
    # Step 4: Generate sparse vectors
    console.print("\n[bold]Step 3:[/bold] Generating sparse vectors (TF-IDF)...")
    vectorizer, tfidf_matrix = generate_sparse_vectors(papers)
    save_vectorizer(vectorizer)
    
    # Step 5: Setup Endee
    console.print("\n[bold]Step 4:[/bold] Setting up Endee connection...")
    client = setup_endee_client()
    
    # Step 6: Create indexes
    console.print("\n[bold]Step 5:[/bold] Creating indexes...")
    create_indexes(client)
    
    # Step 7: Ingest data
    console.print("\n[bold]Step 6:[/bold] Ingesting data into semantic index...")
    ingest_semantic_index(client, papers, embeddings)
    
    console.print("\n[bold]Step 7:[/bold] Ingesting data into hybrid index...")
    ingest_hybrid_index(client, papers, embeddings, tfidf_matrix)
    
    # Step 8: Print stats
    console.print()
    print_index_stats(client)
    
    elapsed = time.time() - start_time
    console.print(f"\n[bold green]✓ Ingestion complete![/bold green] ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
