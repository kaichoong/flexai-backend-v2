"""
Flex AI — Agent definitions (Groq via OpenAI-compatible API)
Using LLaMA 3.3 70B on Groq for fast, free inference.
Phase 2: Critic agent with smart threshold-based retry.
"""

import json
import re
import os
from openai import AsyncOpenAI


def get_client():
    return AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )


def parse_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


async def call_gemini(system: str, user: str, max_tokens: int = 1000) -> dict:
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        raw = response.choices[0].message.content
        print(f"[LLM RAW] {raw[:200]}...")
        return parse_json(raw)
    except Exception as e:
        print(f"[LLM ERROR] {str(e)}")
        return {}


async def call_gemini_text(system: str, user: str, max_tokens: int = 500) -> str:
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[LLM ERROR] {str(e)}")
        return f"Error: {str(e)}"


async def planner_agent(state: dict) -> dict:
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
    user = f"Problem: {state['problem']}\nBudget: ${state['budget']}\n\nAnalyse this and produce a structured brief."
    result = await call_gemini(system, user, 600)
    if not result:
        return {**state, "planner": {}, "log": state.get("log", []) + ["🧠 Planner: ⚠️ failed to get response"], "error": "Planner returned empty"}
    return {**state, "planner": result, "log": state.get("log", []) + ["🧠 Planner: brief defined — " + result.get("scope", "done")]}


async def stack_scout_agent(state: dict) -> dict:
    planner = state.get("planner") or {}
    system = """You are the Stack Scout agent in Flex AI. Respond ONLY with valid JSON, no markdown:
{
  "solutions": [
    {
      "title": "ProductName",
      "type": "software | hardware | ai",
      "stack": ["tool1", "tool2", "tool3"],
      "justification": "why this stack",
      "difficulty": "beginner | intermediate | advanced",
      "tags": ["tag1", "tag2", "tag3"],
      "prerequisites": ["prereq1"],
      "gotchas": ["gotcha1"]
    }
  ]
}
Exactly 3 solutions with product-style titles."""
    user = f"Scope: {planner.get('scope','')}\nType: {planner.get('problem_type','')}\nApproaches: {json.dumps(planner.get('approaches',[]))}\nBudget: ${state['budget']}\n\nIdentify best tech stack for 3 solutions."
    result = await call_gemini(system, user, 800)
    if not result:
        return {**state, "stack_scout": {}, "log": state.get("log",[]) + ["🔍 Stack Scout: ⚠️ failed to get response"]}
    titles = [s.get("title","") for s in result.get("solutions",[])]
    return {**state, "stack_scout": result, "log": state.get("log",[]) + [f"🔍 Stack Scout: {', '.join(titles)} — stacks identified"]}


async def budget_bot_agent(state: dict) -> dict:
    stack_scout = state.get("stack_scout") or {}
    summary = "\n".join([f"{i+1}. {s.get('title','')} ({s.get('type','')}): {', '.join(s.get('stack',[]))}" for i,s in enumerate(stack_scout.get("solutions",[]))])
    system = """You are the Budget Bot agent in Flex AI. Respond ONLY with valid JSON, no markdown:
{
  "solutions": [
    {
      "title": "...",
      "estimated_cost": "Free | $X one-time | $X/mo",
      "breakdown": [{"item": "...", "cost": "Free | $X", "note": "..."}],
      "total_one_time": "$X or Free",
      "total_monthly": "$X/mo or Free",
      "free_alternative": "description or null",
      "within_budget": true
    }
  ]
}"""
    user = f"Budget: ${state['budget']}\n\nStacks:\n{summary}\n\nCalculate realistic costs. Prefer free options."
    result = await call_gemini(system, user, 700)
    if not result:
        return {**state, "budget_bot": {}, "log": state.get("log",[]) + ["💰 Budget Bot: ⚠️ failed to get response"]}
    cost_summary = " | ".join([f"{s.get('title','')}: {s.get('estimated_cost','?')}" for s in result.get("solutions",[])])
    return {**state, "budget_bot": result, "log": state.get("log",[]) + [f"💰 Budget Bot: {cost_summary}"]}


async def tutorial_agent(state: dict, feedback: str = "") -> dict:
    planner = state.get("planner") or {}
    stack_scout = state.get("stack_scout") or {}
    solutions_detail = "\n".join([f"{i+1}. {s.get('title','')} — Stack: {', '.join(s.get('stack',[]))} — Difficulty: {s.get('difficulty','')}" for i,s in enumerate(stack_scout.get("solutions",[]))])
    feedback_block = f"\n\nCRITIC FEEDBACK TO ADDRESS:\n{feedback}" if feedback else ""
    system = """You are the Tutorial Agent in Flex AI. Respond ONLY with valid JSON, no markdown:
{
  "solutions": [
    {
      "title": "...",
      "description": "2 sentences explaining how this solves the user problem",
      "tagline": "one punchy line",
      "phases": [{"phase": "title", "duration": "X days", "steps": ["step1", "step2", "step3"]}],
      "estimated_total_time": "X hours",
      "best_for": "who this suits best"
    }
  ]
}"""
    user = f"Problem: {planner.get('scope', state['problem'])}\nBudget: ${state['budget']}\n\nSolutions:\n{solutions_detail}\n\nWrite descriptions and 2-3 phases per solution.{feedback_block}"
    result = await call_gemini(system, user, 1000)
    if not result:
        return {**state, "tutorial": {}, "log": state.get("log",[]) + ["📋 Tutorial Agent: ⚠️ failed to get response"]}
    suffix = " (revised)" if feedback else ""
    return {**state, "tutorial": result, "log": state.get("log",[]) + [f"📋 Tutorial Agent: build phases written{suffix}"]}


async def code_agent(state: dict, feedback: str = "") -> dict:
    planner = state.get("planner") or {}
    stack_scout = state.get("stack_scout") or {}
    solutions_detail = "\n".join([f"{i+1}. {s.get('title','')} — Stack: {', '.join(s.get('stack',[]))}" for i,s in enumerate(stack_scout.get("solutions",[]))])
    feedback_block = f"\n\nCRITIC FEEDBACK TO ADDRESS:\n{feedback}" if feedback else ""
    system = """You are the Code Agent in Flex AI. Write REAL runnable code only. Respond ONLY with valid JSON, no markdown:
{
  "snippets": [
    {
      "title": "...",
      "filename": "main.py",
      "lang": "python | javascript | bash",
      "install": "pip install X Y",
      "code": "actual working code 10-15 lines",
      "what_it_does": "one sentence"
    }
  ]
}"""
    user = f"Problem: {planner.get('scope', state['problem'])}\n\nSolutions:\n{solutions_detail}\n\nWrite a real starter snippet for each.{feedback_block}"
    result = await call_gemini(system, user, 900)
    if not result:
        return {**state, "code_agent": {}, "log": state.get("log",[]) + ["🤖 Code Agent: ⚠️ failed to get response"]}
    suffix = " (revised)" if feedback else ""
    return {**state, "code_agent": result, "log": state.get("log",[]) + [f"🤖 Code Agent: starter snippets generated{suffix}"]}


async def tools_sourcer_agent(state: dict, feedback: str = "") -> dict:
    stack_scout = state.get("stack_scout") or {}
    summary = "\n".join([f"{i+1}. {s.get('title','')} — Stack: {', '.join(s.get('stack',[]))}" for i,s in enumerate(stack_scout.get("solutions",[]))])
    feedback_block = f"\n\nCRITIC FEEDBACK TO ADDRESS:\n{feedback}" if feedback else ""
    system = """You are the Tools Sourcer agent in Flex AI. Real URLs only. Respond ONLY with valid JSON, no markdown:
{
  "solutions": [
    {
      "title": "...",
      "tools": [{"name": "...", "url": "https://real-url.com", "free": true, "category": "docs | tutorial | tool", "note": "why useful"}]
    }
  ]
}
3-4 tools per solution."""
    user = f"Solutions:\n{summary}\n\nFind real resources for each.{feedback_block}"
    result = await call_gemini(system, user, 700)
    if not result:
        return {**state, "tools_sourcer": {}, "log": state.get("log",[]) + ["📦 Tools Sourcer: ⚠️ failed to get response"]}
    suffix = " (revised)" if feedback else ""
    return {**state, "tools_sourcer": result, "log": state.get("log",[]) + [f"📦 Tools Sourcer: resources found{suffix}"]}


async def critic_agent(state: dict) -> dict:
    """
    Phase 2: Critic agent.
    Scores Tutorial, Code, and Tools outputs 1-10.
    Only retries agents that score below 7.
    """
    planner = state.get("planner") or {}
    tutorial = state.get("tutorial") or {}
    code_agent_out = state.get("code_agent") or {}
    tools_sourcer = state.get("tools_sourcer") or {}

    system = """You are the Critic agent in Flex AI. Review agent outputs and score them.
Respond ONLY with valid JSON, no markdown:
{
  "tutorial_score": 8,
  "tutorial_feedback": "specific actionable feedback, or null if score >= 7",
  "code_score": 7,
  "code_feedback": "specific actionable feedback, or null if score >= 7",
  "tools_score": 6,
  "tools_feedback": "specific actionable feedback, or null if score >= 7"
}
Scoring criteria:
- Tutorial (1-10): Are descriptions specific to the problem? Are phases realistic with enough steps?
- Code (1-10): Is the code real and runnable, not pseudocode? Does it actually match the stack?
- Tools (1-10): Are URLs real and specific (not generic homepages)? Are tools actually relevant?
Score 7+ = acceptable. Below 7 = provide specific, actionable feedback to fix it."""

    user = f"""Problem: {planner.get('scope', state.get('problem', ''))}
Budget: ${state.get('budget', 0)}

TUTORIAL OUTPUT:
{json.dumps(tutorial, indent=2)[:600]}

CODE OUTPUT:
{json.dumps(code_agent_out, indent=2)[:600]}

TOOLS OUTPUT:
{json.dumps(tools_sourcer, indent=2)[:600]}

Score each and give feedback for any scoring below 7."""

    result = await call_gemini(system, user, 600)

    if not result:
        # Critic failed — default all pass so pipeline continues
        print("[CRITIC] Failed to score — defaulting all to pass")
        return {**state, "critic": {"tutorial_score": 7, "code_score": 7, "tools_score": 7}, "log": state.get("log", []) + ["🔎 Critic: ⚠️ scoring failed — skipping retry"]}

    t_score = result.get("tutorial_score", 7)
    c_score = result.get("code_score", 7)
    ts_score = result.get("tools_score", 7)

    score_summary = f"Tutorial:{t_score}/10 Code:{c_score}/10 Tools:{ts_score}/10"
    retries_needed = [k for k, v in [("tutorial", t_score), ("code", c_score), ("tools", ts_score)] if v < 7]

    log_msg = f"🔎 Critic: {score_summary}"
    if retries_needed:
        log_msg += f" — retrying: {', '.join(retries_needed)}"
    else:
        log_msg += " — all passed ✓"

    print(f"[CRITIC] {score_summary} | Retrying: {retries_needed}")
    return {**state, "critic": result, "log": state.get("log", []) + [log_msg]}


async def video_agent(state: dict) -> dict:
    stack_scout = state.get("stack_scout") or {}
    planner = state.get("planner") or {}
    queued = [{"title": s.get("title",""), "type": s.get("type",""), "stack": s.get("stack",[]), "problem_scope": planner.get("scope", state["problem"]), "status": "queued"} for s in stack_scout.get("solutions",[])]
    return {**state, "video_agent": {"queued": queued, "status": "ready"}, "log": state.get("log",[]) + [f"🎬 Video Agent: {len(queued)} video queues ready"]}


async def orchestrator_agent(state: dict) -> dict:
    """
    Phase 3: Full Orchestrator.
    Reads the problem and decides:
    - problem_type (software/hardware/ai/hybrid)
    - solution_count (1-3)
    - agent_order (which agents to run and in what sequence)
    - focus (token/effort allocation per agent)
    - skip (agents to skip entirely)
    - rationale (why these decisions)
    """
    system = """You are the Orchestrator agent in Flex AI. You read the user's problem and design the optimal agent pipeline.
Respond ONLY with valid JSON, no markdown:
{
  "problem_type": "software | hardware | ai | hybrid",
  "complexity": "low | medium | high",
  "solution_count": 2,
  "focus": {
    "planner": "medium",
    "stack_scout": "high",
    "budget_bot": "medium",
    "tutorial": "high",
    "code_agent": "high",
    "tools_sourcer": "low"
  },
  "skip": [],
  "parallel_batch": ["tutorial", "code_agent", "tools_sourcer"],
  "rationale": "one sentence explaining the routing decision",
  "boost_hints": {
    "stack_scout": "focus on hardware components and microcontrollers",
    "code_agent": "prioritise C++ and Python for embedded systems",
    "tutorial": "include wiring diagrams and physical setup steps"
  }
}

Rules:
- solution_count: 1 for very specific problems, 2 for medium, 3 for broad/exploratory
- skip tools_sourcer if problem is very niche or highly specific
- skip budget_bot only if user explicitly said cost doesn't matter
- focus values: "low" | "medium" | "high" | "critical"
- boost_hints: specific instructions to pass to each agent (empty {} if no hints)
- parallel_batch: always include tutorial, code_agent, tools_sourcer unless one is skipped"""

    user = f"""Problem: {state['problem']}
Budget: ${state['budget']}

Analyse this problem and design the optimal agent pipeline."""

    result = await call_gemini(system, user, 700)

    if not result:
        # Safe default — run everything normally
        default_plan = {
            "problem_type": "software",
            "complexity": "medium",
            "solution_count": 3,
            "focus": {"planner": "medium", "stack_scout": "high", "budget_bot": "medium", "tutorial": "high", "code_agent": "high", "tools_sourcer": "medium"},
            "skip": [],
            "parallel_batch": ["tutorial", "code_agent", "tools_sourcer"],
            "rationale": "Default pipeline — orchestrator could not analyse problem",
            "boost_hints": {}
        }
        return {**state, "orchestrator": default_plan, "log": state.get("log", []) + ["🎯 Orchestrator: ⚠️ using default pipeline"]}

    solution_count = result.get("solution_count", 3)
    problem_type = result.get("problem_type", "software")
    skipped = result.get("skip", [])
    rationale = result.get("rationale", "")

    log_msg = f"🎯 Orchestrator: {problem_type} problem → {solution_count} solution(s)"
    if skipped:
        log_msg += f" | skipping: {', '.join(skipped)}"
    log_msg += f" | {rationale}"

    print(f"[ORCHESTRATOR] {log_msg}")
    return {**state, "orchestrator": result, "log": state.get("log", []) + [log_msg]}
