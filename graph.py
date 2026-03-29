"""
Flex AI — LangGraph agent graph
Phase 3: Full Orchestrator intelligence.
Pipeline: Orchestrator → Planner → Stack Scout → [Budget Bot?]
          → Dynamic Parallel(Tutorial+Code+Tools) → Critic+Retry
          → Video Agent → Synthesis
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import asyncio
from agents import (
    orchestrator_agent,
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

# Token multipliers per focus level
FOCUS_TOKENS = {
    "low": 0.6,
    "medium": 1.0,
    "high": 1.4,
    "critical": 1.8,
}


class FlexAIState(TypedDict):
    problem: str
    budget: int
    orchestrator: Optional[dict]
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


def get_focus_tokens(base: int, agent_name: str, orchestrator: dict) -> int:
    """Scale token budget based on orchestrator focus level."""
    focus = orchestrator.get("focus", {})
    level = focus.get(agent_name, "medium")
    multiplier = FOCUS_TOKENS.get(level, 1.0)
    return int(base * multiplier)


def get_boost_hint(agent_name: str, orchestrator: dict) -> str:
    """Get any boost hint for this agent from the orchestrator."""
    hints = orchestrator.get("boost_hints", {})
    return hints.get(agent_name, "")


async def orchestrated_planner(state: FlexAIState) -> FlexAIState:
    """Planner with orchestrator-aware token scaling and hints."""
    orch = state.get("orchestrator") or {}
    hint = get_boost_hint("planner", orch)

    from agents import call_gemini, parse_json
    import json

    system = """You are the Planner agent in Flex AI. Respond ONLY with valid JSON, no markdown:
{
  "scope": "one sentence defining exact scope",
  "success_criteria": ["criterion 1", "criterion 2", "criterion 3"],
  "constraints": ["constraint 1", "constraint 2"],
  "problem_type": "software | hardware | ai | hybrid",
  "complexity": "low | medium | high",
  "approaches": ["approach 1 one sentence", "approach 2 one sentence", "approach 3 one sentence"],
  "target_user": "brief description"
}"""

    solution_count = orch.get("solution_count", 3)
    hint_block = f"\n\nFOCUS HINT: {hint}" if hint else ""
    user = f"Problem: {state['problem']}\nBudget: ${state['budget']}\nSolutions needed: {solution_count}\n\nAnalyse this and produce a structured brief with exactly {solution_count} approaches.{hint_block}"

    max_tokens = get_focus_tokens(600, "planner", orch)
    result = await call_gemini(system, user, max_tokens)

    if not result:
        return {**state, "planner": {}, "log": state.get("log", []) + ["🧠 Planner: ⚠️ failed"], "error": "Planner returned empty"}
    return {**state, "planner": result, "log": state.get("log", []) + ["🧠 Planner: brief defined — " + result.get("scope", "done")]}


async def orchestrated_stack_scout(state: FlexAIState) -> FlexAIState:
    """Stack Scout with orchestrator-aware token scaling and hints."""
    orch = state.get("orchestrator") or {}
    planner = state.get("planner") or {}
    hint = get_boost_hint("stack_scout", orch)
    solution_count = orch.get("solution_count", 3)

    from agents import call_gemini
    import json

    system = f"""You are the Stack Scout agent in Flex AI. Respond ONLY with valid JSON, no markdown:
{{
  "solutions": [
    {{
      "title": "ProductName",
      "type": "software | hardware | ai",
      "stack": ["tool1", "tool2", "tool3"],
      "justification": "why this stack",
      "difficulty": "beginner | intermediate | advanced",
      "tags": ["tag1", "tag2", "tag3"],
      "prerequisites": ["prereq1"],
      "gotchas": ["gotcha1"]
    }}
  ]
}}
Exactly {solution_count} solutions with product-style titles."""

    hint_block = f"\n\nFOCUS HINT: {hint}" if hint else ""
    user = f"Scope: {planner.get('scope','')}\nType: {planner.get('problem_type','')}\nApproaches: {json.dumps(planner.get('approaches',[]))}\nBudget: ${state['budget']}\n\nIdentify best tech stack for {solution_count} solutions.{hint_block}"

    max_tokens = get_focus_tokens(800, "stack_scout", orch)
    result = await call_gemini(system, user, max_tokens)

    if not result:
        return {**state, "stack_scout": {}, "log": state.get("log",[]) + ["🔍 Stack Scout: ⚠️ failed"]}
    titles = [s.get("title","") for s in result.get("solutions",[])]
    return {**state, "stack_scout": result, "log": state.get("log",[]) + [f"🔍 Stack Scout: {', '.join(titles)} — stacks identified"]}


async def orchestrated_budget_bot(state: FlexAIState) -> FlexAIState:
    """Budget Bot — skipped if orchestrator says so."""
    orch = state.get("orchestrator") or {}
    skip = orch.get("skip", [])

    if "budget_bot" in skip:
        return {**state, "budget_bot": {}, "log": state.get("log",[]) + ["💰 Budget Bot: skipped by orchestrator"]}

    return await budget_bot_agent(state)


async def dynamic_parallel(state: FlexAIState) -> FlexAIState:
    """
    Phase 3: Orchestrator-directed parallel execution.
    Runs only agents in parallel_batch, with focus-scaled tokens and boost hints.
    """
    orch = state.get("orchestrator") or {}
    parallel_batch = orch.get("parallel_batch", ["tutorial", "code_agent", "tools_sourcer"])
    skip = orch.get("skip", [])

    tasks = []
    task_keys = []

    if "tutorial" in parallel_batch and "tutorial" not in skip:
        hint = get_boost_hint("tutorial", orch)
        tasks.append(tutorial_agent(state, feedback=hint))
        task_keys.append("tutorial")

    if "code_agent" in parallel_batch and "code_agent" not in skip:
        hint = get_boost_hint("code_agent", orch)
        tasks.append(code_agent(state, feedback=hint))
        task_keys.append("code_agent")

    if "tools_sourcer" in parallel_batch and "tools_sourcer" not in skip:
        hint = get_boost_hint("tools_sourcer", orch)
        tasks.append(tools_sourcer_agent(state, feedback=hint))
        task_keys.append("tools_sourcer")

    skipped = [k for k in ["tutorial", "code_agent", "tools_sourcer"] if k in skip]
    if skipped:
        state = {**state, "log": state.get("log", []) + [f"⚡ Dynamic pipeline: skipping {', '.join(skipped)}"]}

    if not tasks:
        return state

    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged = {**state}
    for key, result in zip(task_keys, results):
        if isinstance(result, Exception):
            merged["log"] = merged.get("log", []) + [f"⚠️ {key} error: {str(result)}"]
        elif isinstance(result, dict):
            merged[key] = result.get(key, {})
            merged["log"] = merged.get("log", []) + result.get("log", [])

    return merged


async def critique_and_retry(state: FlexAIState) -> FlexAIState:
    """Phase 2+3: Critic scores outputs, selectively retries below-threshold agents."""
    orch = state.get("orchestrator") or {}
    skip = orch.get("skip", [])

    state = await critic_agent(state)
    critic = state.get("critic") or {}

    t_score = critic.get("tutorial_score", 7)
    c_score = critic.get("code_score", 7)
    ts_score = critic.get("tools_score", 7)

    retry_tasks = []
    retry_keys = []

    if "tutorial" not in skip and t_score < CRITIC_THRESHOLD and critic.get("tutorial_feedback"):
        retry_tasks.append(tutorial_agent(state, feedback=critic["tutorial_feedback"]))
        retry_keys.append("tutorial")

    if "code_agent" not in skip and c_score < CRITIC_THRESHOLD and critic.get("code_feedback"):
        retry_tasks.append(code_agent(state, feedback=critic["code_feedback"]))
        retry_keys.append("code_agent")

    if "tools_sourcer" not in skip and ts_score < CRITIC_THRESHOLD and critic.get("tools_feedback"):
        retry_tasks.append(tools_sourcer_agent(state, feedback=critic["tools_feedback"]))
        retry_keys.append("tools_sourcer")

    if not retry_tasks:
        return state

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
    orch = state.get("orchestrator") or {}

    solutions = stack_scout.get("solutions", [])
    tut_solutions = tutorial.get("solutions", [])
    budget_solutions = budget_bot.get("solutions", [])
    code_snippets = code_agent_out.get("snippets", [])
    tool_solutions = tools_sourcer.get("solutions", [])

    # Respect orchestrator solution count
    solution_count = orch.get("solution_count", len(solutions))
    solutions = solutions[:solution_count]

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

    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("planner", orchestrated_planner)
    graph.add_node("stack_scout", orchestrated_stack_scout)
    graph.add_node("budget_bot", orchestrated_budget_bot)
    graph.add_node("dynamic_parallel", dynamic_parallel)
    graph.add_node("critique_and_retry", critique_and_retry)
    graph.add_node("video_agent_node", video_agent)
    graph.add_node("synthesise", synthesise)

    graph.add_edge("orchestrator", "planner")
    graph.add_edge("planner", "stack_scout")
    graph.add_edge("stack_scout", "budget_bot")
    graph.add_edge("budget_bot", "dynamic_parallel")
    graph.add_edge("dynamic_parallel", "critique_and_retry")
    graph.add_edge("critique_and_retry", "video_agent_node")
    graph.add_edge("video_agent_node", "synthesise")
    graph.add_edge("synthesise", END)

    graph.set_entry_point("orchestrator")

    return graph.compile()


flex_graph = build_graph()
