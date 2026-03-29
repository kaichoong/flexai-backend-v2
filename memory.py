"""
Flex AI — Supabase Memory Layer (Phase 4)
Uses raw httpx requests instead of the supabase SDK to avoid auth issues.
"""

import os
import httpx
from collections import Counter

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def get_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def is_configured() -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[MEMORY] SUPABASE_URL or SUPABASE_KEY not set")
        return False
    return True


async def save_run(
    fingerprint: str,
    problem: str,
    picked_title: str,
    picked_type: str,
    picked_stack: list,
    difficulty: str,
    budget: int,
    problem_type: str,
    solution_count: int,
) -> bool:
    """Save a completed project run to Supabase via REST API."""
    if not is_configured():
        return False
    try:
        url = f"{SUPABASE_URL}/rest/v1/flex_runs"
        payload = {
            "fingerprint": fingerprint,
            "problem": problem,
            "picked_title": picked_title,
            "picked_type": picked_type,
            "picked_stack": picked_stack,
            "difficulty": difficulty,
            "budget": budget,
            "problem_type": problem_type,
            "solution_count": solution_count,
        }
        print(f"[MEMORY] Saving to {url}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=get_headers())
        print(f"[MEMORY] Save status: {response.status_code} | body: {response.text[:200]}")
        if response.status_code in (200, 201):
            print(f"[MEMORY] Saved run for {fingerprint}: {picked_title}")
            return True
        else:
            print(f"[MEMORY] Save failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"[MEMORY] Save exception: {type(e).__name__}: {str(e)}")
        return False


async def get_user_history(fingerprint: str, limit: int = 3) -> list:
    """Fetch the last N runs for a user fingerprint via REST API."""
    if not is_configured():
        return []
    try:
        url = f"{SUPABASE_URL}/rest/v1/flex_runs"
        params = {
            "fingerprint": f"eq.{fingerprint}",
            "order": "created_at.desc",
            "limit": str(limit),
            "select": "*",
        }
        headers = {**get_headers(), "Prefer": "return=representation"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json() or []
        else:
            print(f"[MEMORY] Fetch failed: {response.status_code} {response.text}")
            return []
    except Exception as e:
        print(f"[MEMORY] Fetch exception: {type(e).__name__}: {str(e)}")
        return []


def build_memory_context(history: list) -> str:
    """Convert user history into a context string for the orchestrator."""
    if not history:
        return ""
    lines = ["USER MEMORY (past runs — use to personalise routing):"]
    for i, run in enumerate(history):
        stack_str = ", ".join(run.get("picked_stack") or [])
        lines.append(
            f"Run {i+1}: '{run.get('problem','')}' → picked {run.get('picked_title','')} "
            f"({run.get('picked_type','')}) | stack: {stack_str} | "
            f"budget: ${run.get('budget',0)} | difficulty: {run.get('difficulty','')}"
        )
    lines.append("\nUse this history to: prefer familiar stack types, match budget range, match difficulty level.")
    return "\n".join(lines)


def get_user_preferences(history: list) -> dict:
    """Derive preference summary from history."""
    if not history:
        return {}
    types = [r.get("picked_type") for r in history if r.get("picked_type")]
    difficulties = [r.get("difficulty") for r in history if r.get("difficulty")]
    budgets = [r.get("budget") for r in history if r.get("budget")]
    stacks = []
    for r in history:
        stacks.extend(r.get("picked_stack") or [])
    return {
        "preferred_type": Counter(types).most_common(1)[0][0] if types else None,
        "preferred_difficulty": Counter(difficulties).most_common(1)[0][0] if difficulties else None,
        "avg_budget": int(sum(budgets) / len(budgets)) if budgets else None,
        "top_stacks": [s for s, _ in Counter(stacks).most_common(3)],
        "run_count": len(history),
    }
