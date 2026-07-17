"""Unit tests for create_deep_search_agent (no network, no real LLM)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepagents import RubricMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import deep_search_agent.factory as factory_module
from deep_search_agent import (
    DEEP_SEARCH_RUBRIC,
    DeepSearchRubricMiddleware,
    DefaultRubricMiddleware,
)
from deep_search_agent.factory import create_deep_search_agent
from deep_search_agent.subagents import (
    FACT_CHECK_AGENT_NAME,
    FETCH_AGENT_NAME,
    PERSPECTIVE_AGENT_NAME,
    SEARCH_AGENT_NAME,
)


class FakeToolCallingModel(GenericFakeChatModel):
    """Fake chat model that tolerates tool binding (returns itself)."""

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def make_fake_model() -> FakeToolCallingModel:
    def messages() -> Iterator[AIMessage]:
        while True:
            yield AIMessage(content="done")

    return FakeToolCallingModel(messages=messages())


@pytest.fixture
def captured(monkeypatch):
    """Intercept the call to deepagents.create_deep_agent, capturing kwargs."""
    calls: dict = {}

    def fake_create_deep_agent(**kwargs):
        calls.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr(factory_module, "create_deep_agent", fake_create_deep_agent)
    return calls


def test_builds_four_builtin_subagents_by_default(captured):
    create_deep_search_agent(model=make_fake_model())

    names = [agent["name"] for agent in captured["subagents"]]
    assert names == [
        PERSPECTIVE_AGENT_NAME,
        SEARCH_AGENT_NAME,
        FETCH_AGENT_NAME,
        FACT_CHECK_AGENT_NAME,
    ]


def test_enable_perspectives_false_omits_perspective_agent(captured):
    create_deep_search_agent(model=make_fake_model(), enable_perspectives=False)

    names = [agent["name"] for agent in captured["subagents"]]
    assert names == [SEARCH_AGENT_NAME, FETCH_AGENT_NAME, FACT_CHECK_AGENT_NAME]


def test_user_subagents_are_appended(captured):
    rag_agent = {
        "name": "rag-agent",
        "description": "Internal knowledge base retrieval",
        "system_prompt": "Retrieve from the vector store.",
    }

    create_deep_search_agent(model=make_fake_model(), subagents=[rag_agent])

    names = [agent["name"] for agent in captured["subagents"]]
    assert names[-1] == "rag-agent"
    assert len(names) == 5


def test_reserved_subagent_name_raises():
    clash = {"name": SEARCH_AGENT_NAME, "description": "x", "system_prompt": "y"}

    with pytest.raises(ValueError, match="reserved"):
        create_deep_search_agent(model=make_fake_model(), subagents=[clash])


def test_perspective_agent_name_reserved_even_when_disabled():
    clash = {"name": PERSPECTIVE_AGENT_NAME, "description": "x", "system_prompt": "y"}

    with pytest.raises(ValueError, match="reserved"):
        create_deep_search_agent(
            model=make_fake_model(), enable_perspectives=False, subagents=[clash]
        )


def test_rubric_middleware_configured_with_cycles(captured):
    create_deep_search_agent(model=make_fake_model(), max_research_cycles=5)

    middleware = captured["middleware"]
    assert isinstance(middleware[0], DefaultRubricMiddleware)
    assert middleware[0].rubric == DEEP_SEARCH_RUBRIC
    rubric_mw = middleware[1]
    # DeepSearchRubricMiddleware (our untruncated-grader subclass), not the
    # bare deepagents RubricMiddleware (issue #22).
    assert isinstance(rubric_mw, DeepSearchRubricMiddleware)
    assert rubric_mw.max_iterations == 5


def test_on_evaluation_forwarded_to_rubric_middleware(captured):
    def callback(evaluation):  # noqa: ARG001
        pass

    create_deep_search_agent(model=make_fake_model(), on_evaluation=callback)

    rubric_mw = captured["middleware"][1]
    assert isinstance(rubric_mw, DeepSearchRubricMiddleware)
    assert rubric_mw._on_evaluation is callback


def test_on_evaluation_defaults_to_none(captured):
    create_deep_search_agent(model=make_fake_model())

    assert captured["middleware"][1]._on_evaluation is None


def test_custom_rubric_is_injected(captured):
    create_deep_search_agent(model=make_fake_model(), rubric="- my criterion")

    assert captured["middleware"][0].rubric == "- my criterion"


def test_auto_rubric_disabled_skips_injection(captured):
    create_deep_search_agent(model=make_fake_model(), auto_rubric=False)

    middleware = captured["middleware"]
    assert not any(isinstance(mw, DefaultRubricMiddleware) for mw in middleware)
    assert any(isinstance(mw, RubricMiddleware) for mw in middleware)


def test_user_middleware_appended_after_rubric(captured):
    class Sentinel(DefaultRubricMiddleware):
        pass

    sentinel = Sentinel("- x")
    create_deep_search_agent(model=make_fake_model(), middleware=[sentinel])

    assert captured["middleware"][-1] is sentinel


def test_subagents_middleware_injected_into_builtin_subagents(captured):
    class Sentinel(DefaultRubricMiddleware):
        pass

    sentinel = Sentinel("- x")
    create_deep_search_agent(model=make_fake_model(), subagents_middleware=[sentinel])

    for agent in captured["subagents"]:
        assert agent["middleware"] == [sentinel]


def test_subagents_middleware_not_injected_into_user_subagents(captured):
    class Sentinel(DefaultRubricMiddleware):
        pass

    rag_agent = {
        "name": "rag-agent",
        "description": "Internal knowledge base retrieval",
        "system_prompt": "Retrieve from the vector store.",
    }
    sentinel = Sentinel("- x")

    create_deep_search_agent(
        model=make_fake_model(),
        subagents=[rag_agent],
        subagents_middleware=[sentinel],
    )

    user_agent = captured["subagents"][-1]
    assert user_agent["name"] == "rag-agent"
    assert "middleware" not in user_agent


def test_no_subagents_middleware_by_default(captured):
    create_deep_search_agent(model=make_fake_model())

    for agent in captured["subagents"]:
        assert "middleware" not in agent


def test_budgets_are_embedded_in_prompts(captured):
    create_deep_search_agent(
        model=make_fake_model(),
        max_research_cycles=4,
        max_search_results_per_query=7,
        max_urls_to_scrape_per_cycle=2,
    )

    assert "4" in captured["system_prompt"]
    assert "at most\n   2 URLs" not in captured["system_prompt"]  # sanity: formatted
    assert "2 URLs per research cycle" in captured["system_prompt"].replace(
        "\n   ", " "
    )
    search_agent = next(
        a for a in captured["subagents"] if a["name"] == SEARCH_AGENT_NAME
    )
    assert "7 results" in search_agent["system_prompt"].replace("\n  ", " ")


def test_query_variants_embedded_in_search_agent_prompt(captured):
    create_deep_search_agent(model=make_fake_model(), max_query_variants=4)

    search_prompt = next(
        a for a in captured["subagents"] if a["name"] == SEARCH_AGENT_NAME
    )["system_prompt"]
    # The search agent is told how many parallel query variants to generate.
    assert "generate 4 distinct query variants" in search_prompt.replace("\n", " ")


def test_query_variants_default_embedded_in_search_agent_prompt(captured):
    create_deep_search_agent(model=make_fake_model())

    search_prompt = next(
        a for a in captured["subagents"] if a["name"] == SEARCH_AGENT_NAME
    )["system_prompt"]
    assert "generate 3 distinct query variants" in search_prompt.replace("\n", " ")


def test_refinement_instructions_in_orchestrator_prompt(captured):
    create_deep_search_agent(model=make_fake_model())

    prompt = captured["system_prompt"]
    assert "## Refinement cycles" in prompt
    assert "research/gaps.md" in prompt


def test_outline_first_synthesis_in_orchestrator_prompt(captured):
    create_deep_search_agent(model=make_fake_model())

    prompt = captured["system_prompt"]
    # Synthesis is split into outline -> section-by-section -> assembly, each
    # writing/using the report structure and numbered citations.
    assert "report/outline.md" in prompt
    assert "OUTLINE" in prompt
    assert "SYNTHESIZE SECTION BY SECTION" in prompt
    assert "ASSEMBLE" in prompt
    assert "executive summary" in prompt
    assert "Gaps & limitations" in prompt


def test_perspective_step_in_orchestrator_prompt_by_default(captured):
    create_deep_search_agent(model=make_fake_model())

    prompt = captured["system_prompt"]
    assert "EXPLORE PERSPECTIVES" in prompt
    assert "perspective-agent" in prompt
    assert "research/perspectives.md" in prompt
    assert "perspective/question pair" in prompt


def test_perspective_step_omitted_when_disabled(captured):
    create_deep_search_agent(model=make_fake_model(), enable_perspectives=False)

    prompt = captured["system_prompt"]
    assert "EXPLORE PERSPECTIVES" not in prompt
    assert "perspective-agent" not in prompt
    assert "research/perspectives.md" not in prompt


def test_rubric_grades_report_structure():
    # The default rubric must grade structure/organization, not only
    # completeness and traceability.
    assert "executive summary" in DEEP_SEARCH_RUBRIC
    assert "Gaps & limitations" in DEEP_SEARCH_RUBRIC
    assert "numbered bibliography" in DEEP_SEARCH_RUBRIC
    assert "perspective/sub-question" in DEEP_SEARCH_RUBRIC


def test_shared_source_index_in_prompts(captured):
    create_deep_search_agent(model=make_fake_model())

    orchestrator = captured["system_prompt"]
    search_agent = next(
        a for a in captured["subagents"] if a["name"] == SEARCH_AGENT_NAME
    )["system_prompt"]
    fetch_agent = next(
        a for a in captured["subagents"] if a["name"] == FETCH_AGENT_NAME
    )["system_prompt"]

    # The `findings/_sources.md` ledger is referenced by the orchestrator and
    # both sub-agents that write to it, so duplicate work can be avoided.
    for prompt in (orchestrator, search_agent, fetch_agent):
        assert "findings/_sources.md" in prompt

    # Refinement cycles must steer sub-agents toward new domains/sources.
    assert "diversify" in orchestrator


def test_custom_system_prompt_wins(captured):
    create_deep_search_agent(model=make_fake_model(), system_prompt="my prompt")

    assert captured["system_prompt"] == "my prompt"


def test_extra_search_tools_reach_search_and_fact_check_agents(captured):
    from langchain_core.tools import tool

    @tool
    def my_kb_search(query: str) -> str:
        """Search the internal knowledge base."""
        return "kb result"

    create_deep_search_agent(model=make_fake_model(), search_tools=[my_kb_search])

    search_agent = next(
        a for a in captured["subagents"] if a["name"] == SEARCH_AGENT_NAME
    )
    fact_check_agent = next(
        a for a in captured["subagents"] if a["name"] == FACT_CHECK_AGENT_NAME
    )
    assert my_kb_search in search_agent["tools"]
    assert my_kb_search in fact_check_agent["tools"]
    # SearxNG tool is always first
    assert search_agent["tools"][0].name == "internet_search"


def test_extra_search_tools_reach_perspective_agent(captured):
    from langchain_core.tools import tool

    @tool
    def my_kb_search(query: str) -> str:
        """Search the internal knowledge base."""
        return "kb result"

    create_deep_search_agent(model=make_fake_model(), search_tools=[my_kb_search])

    perspective_agent = next(
        a for a in captured["subagents"] if a["name"] == PERSPECTIVE_AGENT_NAME
    )
    assert my_kb_search in perspective_agent["tools"]


def test_kwargs_passed_through_to_create_deep_agent(captured):
    create_deep_search_agent(model=make_fake_model(), name="my-researcher", debug=True)

    assert captured["name"] == "my-researcher"
    assert captured["debug"] is True


def test_default_backend_is_a_shared_state_backend(captured):
    from deepagents.backends import StateBackend

    create_deep_search_agent(model=make_fake_model())

    assert isinstance(captured["backend"], StateBackend)


def test_explicit_backend_is_propagated(captured):
    from deepagents.backends import StateBackend

    my_backend = StateBackend()
    create_deep_search_agent(model=make_fake_model(), backend=my_backend)

    assert captured["backend"] is my_backend


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_research_cycles": 0},
        {"max_research_cycles": -1},
        {"max_query_variants": 0},
        {"max_query_variants": -2},
        {"max_search_results_per_query": 0},
        {"max_urls_to_scrape_per_cycle": 0},
        {"searxng_budget": 0},
        {"searxng_budget": -3},
    ],
)
def test_invalid_budgets_raise(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        create_deep_search_agent(model=make_fake_model(), **kwargs)


@pytest.mark.parametrize("rate_limit", [0, -0.5])
def test_invalid_searxng_rate_limit_raises(rate_limit):
    with pytest.raises(ValueError, match="positive number"):
        create_deep_search_agent(model=make_fake_model(), searxng_rate_limit=rate_limit)


@pytest.mark.parametrize("timeout", [0, -5.0])
def test_invalid_js_render_timeout_raises(timeout):
    with pytest.raises(ValueError, match="positive number"):
        create_deep_search_agent(
            model=make_fake_model(),
            enable_js_render_fallback=True,
            js_render_timeout=timeout,
        )


def test_js_render_timeout_unvalidated_while_fallback_is_off(captured):
    # The value is never read when the fallback is off, so it must not gate
    # agent creation for callers that leave the default path alone.
    create_deep_search_agent(model=make_fake_model(), js_render_timeout=0)

    assert captured["subagents"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (False, 30.0)),
        ({"enable_js_render_fallback": True, "js_render_timeout": 45.0}, (True, 45.0)),
    ],
)
def test_js_render_settings_reach_the_fetch_tool(
    monkeypatch, captured, kwargs, expected
):
    from langchain_core.tools import tool

    @tool
    def stub_fetch(url: str) -> str:
        """Stand-in for the real fetch tool."""
        return "content"

    seen: dict = {}

    def fake_create_fetch_url_tool(**tool_kwargs):
        seen.update(tool_kwargs)
        return stub_fetch

    monkeypatch.setattr(
        factory_module, "create_fetch_url_tool", fake_create_fetch_url_tool
    )

    create_deep_search_agent(model=make_fake_model(), **kwargs)

    assert (seen["enable_js_render_fallback"], seen["js_render_timeout"]) == expected


def test_searxng_budget_adds_reset_middleware(captured):
    from deep_search_agent import SearchBudgetResetMiddleware

    create_deep_search_agent(model=make_fake_model(), searxng_budget=5)

    assert any(
        isinstance(mw, SearchBudgetResetMiddleware) for mw in captured["middleware"]
    )


def test_no_budget_middleware_by_default(captured):
    from deep_search_agent import SearchBudgetResetMiddleware

    create_deep_search_agent(model=make_fake_model())

    assert not any(
        isinstance(mw, SearchBudgetResetMiddleware) for mw in captured["middleware"]
    )


def test_reset_middleware_follows_rubric_middleware(captured):
    from deep_search_agent import SearchBudgetResetMiddleware

    create_deep_search_agent(model=make_fake_model(), searxng_budget=5)

    middleware = captured["middleware"]
    rubric_idx = next(
        i for i, mw in enumerate(middleware) if isinstance(mw, RubricMiddleware)
    )
    reset_idx = next(
        i
        for i, mw in enumerate(middleware)
        if isinstance(mw, SearchBudgetResetMiddleware)
    )
    assert reset_idx > rubric_idx


def test_missing_model_raises():
    with pytest.raises(ValueError, match="requires an explicit `model`"):
        create_deep_search_agent(model=None)


# --- metrics wiring ----------------------------------------------------------


def test_metrics_injects_orchestrator_and_subagent_middleware(captured):
    from deep_search_agent import SessionMetrics
    from deep_search_agent.metrics import (
        _OrchestratorMetricsMiddleware,
        _SubagentMetricsMiddleware,
    )

    metrics = SessionMetrics()
    create_deep_search_agent(model=make_fake_model(), metrics=metrics)

    # Orchestrator metrics middleware is present and runs first.
    assert isinstance(captured["middleware"][0], _OrchestratorMetricsMiddleware)
    # Every built-in sub-agent gets its own metrics middleware, named after it.
    for agent in captured["subagents"]:
        metric_mw = [
            mw
            for mw in agent["middleware"]
            if isinstance(mw, _SubagentMetricsMiddleware)
        ]
        assert len(metric_mw) == 1
        assert metric_mw[0]._subagent_name == agent["name"]


def test_metrics_middleware_shares_the_same_collector(captured):
    from deep_search_agent import SessionMetrics
    from deep_search_agent.metrics import (
        _OrchestratorMetricsMiddleware,
        _SubagentMetricsMiddleware,
    )

    metrics = SessionMetrics()
    create_deep_search_agent(model=make_fake_model(), metrics=metrics)

    orchestrator_mw = next(
        mw
        for mw in captured["middleware"]
        if isinstance(mw, _OrchestratorMetricsMiddleware)
    )
    assert orchestrator_mw._metrics is metrics
    for agent in captured["subagents"]:
        for mw in agent["middleware"]:
            if isinstance(mw, _SubagentMetricsMiddleware):
                assert mw._metrics is metrics


def test_metrics_middleware_appended_after_user_subagents_middleware(captured):
    from deep_search_agent import SessionMetrics
    from deep_search_agent.metrics import _SubagentMetricsMiddleware

    sentinel = DefaultRubricMiddleware("- x")
    create_deep_search_agent(
        model=make_fake_model(),
        metrics=SessionMetrics(),
        subagents_middleware=[sentinel],
    )

    for agent in captured["subagents"]:
        mws = agent["middleware"]
        assert mws[0] is sentinel
        assert isinstance(mws[-1], _SubagentMetricsMiddleware)


def test_no_metrics_middleware_by_default(captured):
    from deep_search_agent.metrics import (
        _OrchestratorMetricsMiddleware,
        _SubagentMetricsMiddleware,
    )

    create_deep_search_agent(model=make_fake_model())

    assert not any(
        isinstance(mw, _OrchestratorMetricsMiddleware) for mw in captured["middleware"]
    )
    for agent in captured["subagents"]:
        assert "middleware" not in agent or not any(
            isinstance(mw, _SubagentMetricsMiddleware) for mw in agent["middleware"]
        )


def test_end_to_end_graph_compiles():
    """Smoke test against the real create_deep_agent (no LLM calls)."""
    agent = create_deep_search_agent(model=make_fake_model())

    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")
