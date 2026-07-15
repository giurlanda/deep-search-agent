"""Unit tests for create_deep_search_agent (no network, no real LLM)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepagents import RubricMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import deep_search_agent.factory as factory_module
from deep_search_agent import DEEP_SEARCH_RUBRIC, DefaultRubricMiddleware
from deep_search_agent.factory import create_deep_search_agent
from deep_search_agent.subagents import (
    FACT_CHECK_AGENT_NAME,
    FETCH_AGENT_NAME,
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


def test_builds_three_builtin_subagents(captured):
    create_deep_search_agent(model=make_fake_model())

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
    assert len(names) == 4


def test_reserved_subagent_name_raises():
    clash = {"name": SEARCH_AGENT_NAME, "description": "x", "system_prompt": "y"}

    with pytest.raises(ValueError, match="reserved"):
        create_deep_search_agent(model=make_fake_model(), subagents=[clash])


def test_rubric_middleware_configured_with_cycles(captured):
    create_deep_search_agent(model=make_fake_model(), max_research_cycles=5)

    middleware = captured["middleware"]
    assert isinstance(middleware[0], DefaultRubricMiddleware)
    assert middleware[0].rubric == DEEP_SEARCH_RUBRIC
    rubric_mw = middleware[1]
    assert isinstance(rubric_mw, RubricMiddleware)
    assert rubric_mw.max_iterations == 5


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
    search_agent = captured["subagents"][0]
    assert "7 results" in search_agent["system_prompt"].replace("\n  ", " ")


def test_refinement_instructions_in_orchestrator_prompt(captured):
    create_deep_search_agent(model=make_fake_model())

    prompt = captured["system_prompt"]
    assert "## Refinement cycles" in prompt
    assert "research/gaps.md" in prompt


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

    search_agent = captured["subagents"][0]
    fact_check_agent = captured["subagents"][2]
    assert my_kb_search in search_agent["tools"]
    assert my_kb_search in fact_check_agent["tools"]
    # SearxNG tool is always first
    assert search_agent["tools"][0].name == "internet_search"


def test_kwargs_passed_through_to_create_deep_agent(captured):
    create_deep_search_agent(model=make_fake_model(), name="my-researcher", debug=True)

    assert captured["name"] == "my-researcher"
    assert captured["debug"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_research_cycles": 0},
        {"max_research_cycles": -1},
        {"max_search_results_per_query": 0},
        {"max_urls_to_scrape_per_cycle": 0},
    ],
)
def test_invalid_budgets_raise(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        create_deep_search_agent(model=make_fake_model(), **kwargs)


def test_missing_model_raises():
    with pytest.raises(ValueError, match="requires an explicit `model`"):
        create_deep_search_agent(model=None)


def test_end_to_end_graph_compiles():
    """Smoke test against the real create_deep_agent (no LLM calls)."""
    agent = create_deep_search_agent(model=make_fake_model())

    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")
