"""
ScholarMind — Streamlit Web Dashboard
Interactive UI for semantic search, hybrid search, filtered queries, and RAG Q&A.
"""

import streamlit as st
import time

# ── Page Config ──
st.set_page_config(
    page_title="ScholarMind — AI Research Discovery",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: #e8e8ff;
    }
    
    .result-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .score-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .year-badge {
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .category-badge {
        background: rgba(118, 75, 162, 0.15);
        color: #b48ade;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        border: 1px solid rgba(118, 75, 162, 0.3);
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1e1e3a, #252550);
        border: 1px solid #3a3a6a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-card h3 {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    .stat-card p {
        font-size: 0.85rem;
        color: #888;
        margin: 0.3rem 0 0 0;
    }
    
    .rag-answer {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .source-item {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    
    .sidebar .stSelectbox label, .sidebar .stSlider label {
        color: #b0b0d0 !important;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e1e3a, #252550);
        border: 1px solid #3a3a6a;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ──
@st.cache_resource
def load_search_engine():
    """Load and cache the search engine."""
    from search import ScholarSearch
    return ScholarSearch()


@st.cache_resource
def load_rag_engine():
    """Load and cache the RAG engine."""
    try:
        from rag import ScholarRAG
        return ScholarRAG()
    except Exception as e:
        return None


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 ScholarMind</h1>
        <p>AI-Powered Research Paper Discovery & QA Engine — Powered by Endee Vector Database</p>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(result: dict, rank: int):
    """Render a single search result as a styled card."""
    category_display = result.get("category", "").replace("_", " ").title()
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.8rem;">
            <div style="flex: 1;">
                <span style="color: #555; font-size: 0.8rem;">#{rank}</span>
                <h3 style="color: #e0e0ff; font-size: 1.15rem; font-weight: 600; margin: 0.2rem 0;">
                    {result['title']}
                </h3>
            </div>
            <span class="score-badge">{result['similarity']:.4f}</span>
        </div>
        <p style="color: #8888aa; font-size: 0.9rem; margin: 0.3rem 0;">
            ✍️ {result.get('authors', 'Unknown')}
        </p>
        <p style="color: #aaaacc; font-size: 0.9rem; line-height: 1.5; margin: 0.8rem 0;">
            {result.get('abstract', '')[:250]}{'...' if len(result.get('abstract', '')) > 250 else ''}
        </p>
        <div style="display: flex; gap: 0.5rem; margin-top: 0.8rem;">
            <span class="year-badge">📅 {result.get('year', 'N/A')}</span>
            <span class="category-badge">📂 {category_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats(engine):
    """Render index statistics."""
    stats = engine.get_index_stats()
    cols = st.columns(4)
    
    total_vectors = 0
    for name, info in stats.items():
        if isinstance(info, dict) and "vector_count" in info:
            total_vectors += info.get("vector_count", 0)
    
    with cols[0]:
        st.metric("📚 Total Papers Indexed", total_vectors // 2 if total_vectors else "N/A")
    with cols[1]:
        st.metric("🔢 Total Vectors", total_vectors)
    with cols[2]:
        st.metric("📐 Embedding Dimension", "384")
    with cols[3]:
        st.metric("🏗️ Indexes", len(stats))


def main():
    render_header()
    
    # ── Load Engine ──
    try:
        engine = load_search_engine()
    except Exception as e:
        st.error(f"❌ Could not connect to Endee. Make sure the server is running.\n\nError: {e}")
        st.info("💡 Run `docker compose up -d` in the project directory to start Endee.")
        return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        
        tab_mode = st.radio(
            "Mode",
            ["🔍 Search", "🤖 Ask AI (RAG)"],
            horizontal=True,
        )
        
        st.divider()
        
        if tab_mode == "🔍 Search":
            search_mode = st.selectbox(
                "Search Method",
                ["Semantic Search", "Hybrid Search", "Filtered Search", "Multi-Area Search"],
                help="Choose how to search the paper database",
            )
            
            top_k = st.slider("Results to show", 1, 20, 10)
            
            st.divider()
            
            # ── Filters ──
            category_filter = None
            area_filter = None
            areas_filter = []
            year_range = (2013, 2024)
            
            if search_mode in ["Filtered Search", "Multi-Area Search"]:
                st.markdown("### 🎯 Filters")
                
                if search_mode == "Filtered Search":
                    category_filter = st.selectbox(
                        "Category",
                        [None] + sorted([
                            "machine_learning", "natural_language_processing",
                            "computer_vision", "reinforcement_learning",
                            "ai_safety", "generative_ai", "robotics",
                            "graph_neural_networks",
                        ]),
                        format_func=lambda x: "All Categories" if x is None else x.replace("_", " ").title(),
                    )
                    
                    area_filter = st.selectbox(
                        "Research Area",
                        [None] + sorted([
                            "nlp", "cv", "rl", "ml", "ai", "dl",
                            "transformers", "diffusion", "gan",
                            "optimization", "representation_learning",
                        ]),
                        format_func=lambda x: "All Areas" if x is None else x.upper(),
                    )
                    
                    year_range = st.slider(
                        "Year Range",
                        min_value=2013,
                        max_value=2024,
                        value=(2013, 2024),
                    )
                
                elif search_mode == "Multi-Area Search":
                    areas_filter = st.multiselect(
                        "Select Areas",
                        [
                            "nlp", "cv", "rl", "ml", "ai", "dl",
                            "transformers", "diffusion", "gan",
                            "optimization", "representation_learning",
                        ],
                        default=["nlp", "cv"],
                        format_func=lambda x: x.upper(),
                    )
            
            st.divider()
            st.markdown("### 📊 Database Stats")
            try:
                render_stats(engine)
            except Exception:
                st.info("Run ingestion first to see stats.")
        
        else:  # RAG mode
            rag_search_mode = st.selectbox(
                "Retrieval Method",
                ["semantic", "hybrid", "filtered"],
                format_func=lambda x: x.title(),
            )
            rag_top_k = st.slider("Papers to retrieve", 1, 10, 5)
            
            rag_category = None
            rag_year_min = None
            rag_year_max = None
            
            if rag_search_mode == "filtered":
                st.markdown("### 🎯 Filters")
                rag_category = st.selectbox(
                    "Category",
                    [None] + sorted([
                        "machine_learning", "natural_language_processing",
                        "computer_vision", "reinforcement_learning",
                        "ai_safety", "generative_ai",
                    ]),
                    format_func=lambda x: "All" if x is None else x.replace("_", " ").title(),
                    key="rag_cat",
                )
                rag_year_range = st.slider("Year Range", 2013, 2024, (2013, 2024), key="rag_yr")
                rag_year_min, rag_year_max = rag_year_range

    # ── Main Content ──
    if tab_mode == "🔍 Search":
        query = st.text_input(
            "🔎 Search research papers...",
            placeholder="e.g., How do attention mechanisms work in transformers?",
            key="search_query",
        )
        
        if query:
            with st.spinner("Searching Endee..."):
                start = time.time()
                
                try:
                    if search_mode == "Semantic Search":
                        results = engine.semantic_search(query, top_k=top_k)
                    elif search_mode == "Hybrid Search":
                        results = engine.hybrid_search(query, top_k=top_k)
                    elif search_mode == "Filtered Search":
                        results = engine.filtered_search(
                            query,
                            category=category_filter,
                            area=area_filter,
                            year_min=year_range[0],
                            year_max=year_range[1],
                            top_k=top_k,
                        )
                    elif search_mode == "Multi-Area Search":
                        if not areas_filter:
                            st.warning("Please select at least one area in the sidebar.")
                            return
                        results = engine.multi_area_search(query, areas=areas_filter, top_k=top_k)
                    
                    elapsed = time.time() - start
                    
                    st.markdown(f"**{len(results)} results** found in **{elapsed:.3f}s** using **{search_mode}**")
                    st.divider()
                    
                    for i, result in enumerate(results, 1):
                        render_result_card(result, i)
                    
                    if not results:
                        st.info("No results found. Try a different query or broaden your filters.")
                        
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.info("Make sure Endee is running and data has been ingested. Run `python ingest.py` first.")
    
    else:  # RAG Mode
        st.markdown("### 🤖 Ask AI about Research Papers")
        st.markdown("*Powered by Endee retrieval + Gemini generation*")
        
        question = st.text_area(
            "Ask a research question...",
            placeholder="e.g., What are the key differences between GANs and diffusion models for image generation?",
            height=100,
            key="rag_question",
        )
        
        ask_button = st.button("🚀 Ask ScholarMind", type="primary", use_container_width=True)
        
        if ask_button and question:
            rag_engine = load_rag_engine()
            if rag_engine is None:
                st.error("❌ RAG engine not available. Set GEMINI_API_KEY environment variable.")
                return
            
            with st.spinner("🔍 Retrieving papers & generating answer..."):
                try:
                    result = rag_engine.ask(
                        question,
                        search_mode=rag_search_mode,
                        top_k=rag_top_k,
                        category=rag_category if rag_search_mode == "filtered" else None,
                        year_min=rag_year_min if rag_search_mode == "filtered" else None,
                        year_max=rag_year_max if rag_search_mode == "filtered" else None,
                    )
                    
                    # ── Answer ──
                    st.markdown("---")
                    st.markdown("#### 💡 Answer")
                    st.markdown(f"""<div class="rag-answer">{result['answer']}</div>""", unsafe_allow_html=True)
                    
                    # ── Sources ──
                    st.markdown("#### 📚 Sources Retrieved from Endee")
                    for s in result["sources"]:
                        st.markdown(f"""
                        <div class="source-item">
                            <strong>{s['title']}</strong> ({s['year']})
                            <br><span style="color: #888;">By {s['authors']}</span>
                            <span class="score-badge" style="float: right;">{s['similarity']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.caption(f"Retrieval mode: {result['search_mode'].title()}")
                    
                except Exception as e:
                    st.error(f"RAG pipeline error: {e}")


if __name__ == "__main__":
    main()
