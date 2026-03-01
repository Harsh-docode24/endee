"""
ScholarMind — Evaluation & Benchmarking
Measures retrieval quality (Recall@5, MRR@10) and latency across search modes.
"""

import time
import json
import os
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ── Ground-Truth Labeled Queries ─────────────────────────────────────
# Each query has expected relevant paper IDs (human-labeled)
LABELED_QUERIES = [
    {
        "query": "How do transformers process sequences without recurrence?",
        "relevant": ["paper_001", "paper_002", "paper_009"],
    },
    {
        "query": "Pre-training bidirectional language representations",
        "relevant": ["paper_002", "paper_003", "paper_017"],
    },
    {
        "query": "Generative adversarial networks for image synthesis",
        "relevant": ["paper_005", "paper_006", "paper_018"],
    },
    {
        "query": "Deep reinforcement learning for playing Atari games",
        "relevant": ["paper_007", "paper_008", "paper_010"],
    },
    {
        "query": "Residual connections in deep convolutional networks",
        "relevant": ["paper_004", "paper_031", "paper_034"],
    },
    {
        "query": "Retrieval augmented generation for question answering",
        "relevant": ["paper_012", "paper_003", "paper_024"],
    },
    {
        "query": "AI safety and alignment of large language models",
        "relevant": ["paper_013", "paper_014", "paper_030"],
    },
    {
        "query": "Vision transformer for image classification",
        "relevant": ["paper_009", "paper_004", "paper_016"],
    },
    {
        "query": "Diffusion models for high resolution image generation",
        "relevant": ["paper_006", "paper_018", "paper_026"],
    },
    {
        "query": "Policy gradient methods in reinforcement learning",
        "relevant": ["paper_008", "paper_007", "paper_032"],
    },
    {
        "query": "Parameter efficient fine-tuning of large models",
        "relevant": ["paper_025", "paper_002", "paper_023"],
    },
    {
        "query": "Graph neural network architectures and attention",
        "relevant": ["paper_015", "paper_029", "paper_039"],
    },
    {
        "query": "Scaling laws for neural network training",
        "relevant": ["paper_023", "paper_003", "paper_035"],
    },
    {
        "query": "Object detection in real time",
        "relevant": ["paper_011", "paper_016", "paper_004"],
    },
    {
        "query": "Text to video generation with diffusion transformers",
        "relevant": ["paper_045", "paper_006", "paper_018"],
    },
    {
        "query": "Optimizers for training deep neural networks",
        "relevant": ["paper_020", "paper_019", "paper_021"],
    },
    {
        "query": "Prompt engineering and chain of thought reasoning",
        "relevant": ["paper_024", "paper_003", "paper_022"],
    },
    {
        "query": "Robot manipulation learning from demonstrations",
        "relevant": ["paper_037", "paper_036", "paper_007"],
    },
]


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-K results."""
    top_k = retrieved_ids[:k]
    found = len(set(top_k) & set(relevant_ids))
    return found / len(relevant_ids) if relevant_ids else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """MRR@K: reciprocal rank of the first relevant result in top-K."""
    for rank, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_search_mode(engine, mode: str, queries: list[dict], top_k: int = 10):
    """Run evaluation for a specific search mode."""
    recalls_5 = []
    recalls_10 = []
    mrrs_10 = []
    latencies = []

    for q in queries:
        start = time.time()

        if mode == "semantic":
            results = engine.semantic_search(q["query"], top_k=top_k)
        elif mode == "hybrid":
            results = engine.hybrid_search(q["query"], top_k=top_k)
        else:
            results = engine.semantic_search(q["query"], top_k=top_k)

        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved_ids = [r["id"] for r in results]
        relevant_ids = q["relevant"]

        recalls_5.append(recall_at_k(retrieved_ids, relevant_ids, 5))
        recalls_10.append(recall_at_k(retrieved_ids, relevant_ids, 10))
        mrrs_10.append(mrr_at_k(retrieved_ids, relevant_ids, 10))

    return {
        "mode": mode,
        "recall@5": np.mean(recalls_5),
        "recall@10": np.mean(recalls_10),
        "mrr@10": np.mean(mrrs_10),
        "avg_latency_ms": np.mean(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "num_queries": len(queries),
    }


def run_evaluation():
    """Run full evaluation across all search modes."""
    from search import ScholarSearch

    console.print(Panel.fit(
        "[bold cyan]ScholarMind[/bold cyan] — Retrieval Evaluation\n"
        f"Running {len(LABELED_QUERIES)} labeled queries across search modes",
        border_style="bright_blue",
    ))

    engine = ScholarSearch()

    # Evaluate each mode
    results = []
    for mode in ["semantic", "hybrid"]:
        console.print(f"\n[bold]Evaluating {mode} search...[/bold]")
        metrics = evaluate_search_mode(engine, mode, LABELED_QUERIES)
        results.append(metrics)
        console.print(f"  [green]✓[/green] {mode}: Recall@5={metrics['recall@5']:.3f}, MRR@10={metrics['mrr@10']:.3f}, Latency={metrics['avg_latency_ms']:.1f}ms")

    # ── Results Table ──
    console.print()
    table = Table(title="📊 Retrieval Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Search Mode", style="cyan")
    table.add_column("Recall@5", justify="right", style="green")
    table.add_column("Recall@10", justify="right", style="green")
    table.add_column("MRR@10", justify="right", style="yellow")
    table.add_column("Avg Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Queries", justify="right")

    for m in results:
        table.add_row(
            m["mode"].title(),
            f"{m['recall@5']:.3f}",
            f"{m['recall@10']:.3f}",
            f"{m['mrr@10']:.3f}",
            f"{m['avg_latency_ms']:.1f}ms",
            f"{m['p95_latency_ms']:.1f}ms",
            str(m["num_queries"]),
        )

    console.print(table)

    # ── Save results to JSON ──
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]✓[/green] Results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
