from deep_search_agent import create_deep_search_agent
from langchain_openai import ChatOpenAI
from debug_middleware import DebugMiddleware

model_openrouter = ChatOpenAI(
    model="qwen3.6-35b-a3b-ud-mlx",
    base_url="http://127.0.0.1:1234/v1",
    api_key="no_api_key",
    temperature=0.1,
    timeout=120,
    max_retries=1,
    stream_usage=True,
)


agent = create_deep_search_agent(
    model=model_openrouter,
    searxng_base_url="http://localhost:8888",
    middleware=[DebugMiddleware()],
    max_research_cycles=3,
)

QUERY = """Conduct a comprehensive, evidence-based analysis of the current state and near-term trajectory of AI alignment research and governance frameworks. Specifically:
(1) Map the top three technical approaches to alignment (e.g., RLHF, constitutional AI, mechanistic interpretability) and evaluate their documented success rates, failure modes, and scalability limits based on 2023–2024 peer-reviewed studies and open-source benchmarks.
(2) Compare how regulatory frameworks in the EU (AI Act), US (NIST AI RMF + executive orders), and China (generative AI regulations) differ in enforcement mechanisms, compliance costs, and impact on open vs. closed model development.
(3) Identify key economic incentives and market failures that drive or hinder alignment investment, citing venture funding trends, corporate R&D allocation, and academic grant data.
(4) Synthesize conflicting expert forecasts on whether current alignment methods will scale to AGI-level systems, explicitly distinguishing between empirical evidence, theoretical arguments, and speculative projections.
(5) Provide a structured risk matrix ranking the most probable alignment failure modes by 2030, with confidence intervals and source attribution.
Prioritize primary sources: arXiv/NeurIPS/ICML papers, official policy documents, industry whitepapers, and verified funding databases. Flag any claims that lack empirical validation or contradict established benchmarks. Return sources with direct links and publication dates.
"""

result = agent.invoke(
    {"messages": [{"role": "user", "content": QUERY}]},
    config={"configurable": {"thread_id": "ricerca-1"}},
)
print(result["messages"][-1].content)
