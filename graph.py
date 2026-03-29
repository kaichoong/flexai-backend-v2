"""
Flex AI — LangGraph agent graph
Phase 2: Critic agent with smart threshold-based retry.
Pipeline: Planner → Stack Scout → Budget Bot → Parallel(Tutorial+Code+Tools)
         → Critic → [selective retry] → Video Agent → Synthesis
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import asyncio
from agents import (
    planner_agent,
    stack_scout_agent,
    budget_bot_agent,
    tutorial_agent,
    code_agent,
    tools_sourcer_agent,
    critic_agent,
    video_agent,
)

CRITIC_THRESHOLD = 7


class FlexAIState(TypedDict):
    problem: str
    budget: int
    planner: Optional[dict]
    stack_scout: Optional[dict]
    budget_bot: Optional[dict]
    tutorial: Optional[dict]
    code_agent: Optional[dict]
    tools_sourcer: Optional[dict]
    critic: Optional[dict]
    video_agent: Optional[dict]
    log: list
    error: Optional[str]


async def parallel_agents(state: FlexAIState) -> FlexAIState:
    """Run Tutorial, Code, Tools agents in parallel — first pass."""
    results = await asyncio.gather(
        tutorial_agent(state),
        code_agent(state),
        tools_sourcer_agent(state),
        return_exceptions=True
    )

    merged = {**state}
    for result in results:
        if isinstance(result, Exception):
            merged["log"] = merged.get("log", []) + [f"⚠️ Agent error: {str(result)}"]
        elif isinstance(result, dict):
            for key in ["tutorial", "code_agent", "tools_sourcer", "log"]:
                if key in result:
                    if key == "log":
                        merged["log"] = merged.get("log", []) + result.get("log", [])
                    else:
                        merged[key] = result[key]

    return merged


async def critique_and_retry(state: FlexAIState) -> FlexAIState:
    """
    Phase 2: Run critic, then selectively retry only agents that scored below threshold.
    Max 1 retry per agent. All retries run in parallel.
    """
    # Run critic
    state = await critic_agent(state)
    critic = state.get("critic") or {}

    t_score = critic.get("tutorial_score", 7)
    c_score = critic.get("code_score", 7)
    ts_score = critic.get("tools_score", 7)

    # Build list of agents that need retrying
    retry_tasks = []
    retry_keys = []

    if t_score < CRITIC_THRESHOLD and critic.get("tutorial_feedback"):
        retry_tasks.append(tutorial_agent(state, feedback=critic["tutorial_feedback"]))
        retry_keys.append("tutorial")

    if c_score < CRITIC_THRESHOLD and critic.get("code_feedback"):
        retry_tasks.append(code_agent(state, feedback=critic["code_feedback"]))
        retry_keys.append("code_agent")

    if ts_score < CRITIC_THRESHOLD and critic.get("tools_feedback"):
        retry_tasks.append(tools_sourcer_agent(state, feedback=critic["tools_feedback"]))
        retry_keys.append("tools_sourcer")

    if not retry_tasks:
        # All passed — nothing to retry
        return state

    # Run retries in parallel
    retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

    merged = {**state}
    for key, result in zip(retry_keys, retry_results):
        if isinstance(result, Exception):
            merged["log"] = merged.get("log", []) + [f"⚠️ Retry error for {key}: {str(result)}"]
        elif isinstance(result, dict):
            merged[key] = result.get(key, merged.get(key))
            merged["log"] = merged.get("log", []) + result.get("log", [])

    return merged


async def synthesise(state: FlexAIState) -> FlexAIState:
    stack_scout = state.get("stack_scout") or {}
    tutorial = state.get("tutorial") or {}
    budget_bot = state.get("budget_bot") or {}
    code_agent_out = state.get("code_agent") or {}
    tools_sourcer = state.get("tools_sourcer") or {}

    solutions = stack_scout.get("solutions", [])
    tut_solutions = tutorial.get("solutions", [])
    budget_solutions = budget_bot.get("solutions", [])
    code_snippets = code_agent_out.get("snippets", [])
    tool_solutions = tools_sourcer.get("solutions", [])

    projects = []
    for i, sol in enumerate(solutions):
        tut = tut_solutions[i] if i < len(tut_solutions) else {}
        budget = budget_solutions[i] if i < len(budget_solutions) else {}
        code = code_snippets[i] if i < len(code_snippets) else {}
        tools = tool_solutions[i] if i < len(tool_solutions) else {}

        projects.append({
            "title": sol.get("title", ""),
            "tagline": tut.get("tagline", sol.get("justification", "")),
            "type": sol.get("type", "software"),
            "difficulty": sol.get("difficulty", "intermediate"),
            "tags": sol.get("tags", []),
            "description": tut.get("description", ""),
            "best_for": tut.get("best_for", ""),
            "estimated_total_time": tut.get("estimated_total_time", ""),
            "stack": sol.get("stack", []),
            "justification": sol.get("justification", ""),
            "prerequisites": sol.get("prerequisites", []),
            "gotchas": sol.get("gotchas", []),
            "estimated_cost": budget.get("estimated_cost", ""),
            "cost_breakdown": budget.get("breakdown", []),
            "total_monthly": budget.get("total_monthly", ""),
            "free_alternative": budget.get("free_alternative"),
            "within_budget": budget.get("within_budget", True),
            "phases": tut.get("phases", []),
            "starter_code": {
                "filename": code.get("filename", "main.py"),
                "lang": code.get("lang", "python"),
                "install": code.get("install", ""),
                "code": code.get("code", ""),
                "what_it_does": code.get("what_it_does", ""),
            } if code else None,
            "tools": tools.get("tools", []),
        })

    return {
        **state,
        "projects": projects,
        "log": state.get("log", []) + [f"✓ Synthesis complete — {len(projects)} solutions ready"]
    }


def build_graph():
    graph = StateGraph(dict)

    graph.add_node("planner", planner_agent)
    graph.add_node("stack_scout", stack_scout_agent)
    graph.add_node("budget_bot", budget_bot_agent)
    graph.add_node("parallel_agents", parallel_agents)
    graph.add_node("critique_and_retry", critique_and_retry)
    graph.add_node("video_agent_node", video_agent)
    graph.add_node("synthesise", synthesise)

    graph.add_edge("planner", "stack_scout")
    graph.add_edge("stack_scout", "budget_bot")
    graph.add_edge("budget_bot", "parallel_agents")
    graph.add_edge("parallel_agents", "critique_and_retry")
    graph.add_edge("critique_and_retry", "video_agent_node")
    graph.add_edge("video_agent_node", "synthesise")
    graph.add_edge("synthesise", END)

    graph.set_entry_point("planner")

    return graph.compile()


flex_graph = build_graph()
