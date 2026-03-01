"""
ScholarMind — RAG (Retrieval-Augmented Generation) Pipeline
Uses Endee for retrieval + Gemini for generation to answer research questions.
"""

import google.generativeai as genai
from search import ScholarSearch
import config


class ScholarRAG:
    """RAG pipeline: Endee retrieval → Gemini generation."""

    SYSTEM_PROMPT = """You are ScholarMind, an AI research assistant specialized in academic papers.

Your role is to answer questions about AI/ML research by synthesizing information from 
retrieved research papers. Always:

1. Ground your answers in the provided paper abstracts and metadata
2. Cite specific papers by title when referencing information
3. Be precise about claims — distinguish between what papers say vs. your interpretation
4. If the retrieved papers don't contain enough info to answer, say so honestly
5. Use clear, academic but accessible language

Format your response with:
- A direct answer first
- Supporting evidence from specific papers
- Relevant connections between papers if applicable"""

    def __init__(self):
        """Initialize RAG pipeline."""
        self.search = ScholarSearch()
        self._setup_gemini()

    def _setup_gemini(self):
        """Configure Gemini API."""
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Set it via environment variable or in config.py"
            )
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(
            config.GEMINI_MODEL,
            system_instruction=self.SYSTEM_PROMPT,
        )

    def _build_context(self, results: list[dict]) -> str:
        """Build context string from search results."""
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Paper {i}] \"{r['title']}\" ({r['year']})\n"
                f"Authors: {r['authors']}\n"
                f"Category: {r['category']}\n"
                f"Keywords: {r['keywords']}\n"
                f"Abstract: {r['abstract']}\n"
                f"Relevance Score: {r['similarity']}\n"
            )
        return "\n---\n".join(context_parts)

    def ask(
        self,
        question: str,
        search_mode: str = "semantic",
        top_k: int = 5,
        category: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> dict:
        """
        Answer a research question using RAG.
        
        Args:
            question: The research question to answer
            search_mode: "semantic", "hybrid", or "filtered"
            top_k: Number of papers to retrieve for context
            category: Optional category filter (for filtered mode)
            year_min: Optional minimum year (for filtered mode)
            year_max: Optional maximum year (for filtered mode)
            
        Returns:
            dict with 'answer', 'sources', and 'search_mode'
        """
        # Step 1: Retrieve relevant papers from Endee
        if search_mode == "hybrid":
            results = self.search.hybrid_search(question, top_k=top_k)
        elif search_mode == "filtered":
            results = self.search.filtered_search(
                question,
                category=category,
                year_min=year_min,
                year_max=year_max,
                top_k=top_k,
            )
        else:
            results = self.search.semantic_search(question, top_k=top_k)

        if not results:
            return {
                "answer": "I couldn't find any relevant papers to answer your question. Try rephrasing or broadening your search.",
                "sources": [],
                "search_mode": search_mode,
            }

        # Step 2: Build context from retrieved papers
        context = self._build_context(results)

        # Step 3: Generate answer with Gemini
        prompt = (
            f"Based on the following research papers retrieved from the database, "
            f"answer this question:\n\n"
            f"**Question:** {question}\n\n"
            f"**Retrieved Papers:**\n{context}\n\n"
            f"Provide a comprehensive answer grounded in these papers. "
            f"Cite papers by their title when referencing specific information."
        )

        response = self.gemini_model.generate_content(prompt)

        return {
            "answer": response.text,
            "sources": [
                {
                    "title": r["title"],
                    "year": r["year"],
                    "similarity": r["similarity"],
                    "authors": r["authors"],
                }
                for r in results
            ],
            "search_mode": search_mode,
        }


# ── CLI Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]ScholarMind[/bold cyan] — RAG Q&A Demo",
        border_style="bright_blue",
    ))

    rag = ScholarRAG()

    question = "What is the transformer architecture and how has it evolved?"
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    result = rag.ask(question, search_mode="semantic", top_k=5)

    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    console.print("\n[bold]Sources:[/bold]")
    for s in result["sources"]:
        console.print(f"  📄 {s['title']} ({s['year']}) — Score: {s['similarity']}")
