"""
Flex AI — Supabase Memory Layer (Phase 4)
Saves project runs and reads past history to inform the orchestrator.
Anonymous memory via browser fingerprint — no auth required.
"""

import os
import json
from datetime import datetime
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# SQL to create the table (run once in Supabase SQL editor):
# CREATE TABLE flex_runs (
#   id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
#   fingerprint text NOT NULL,
#   problem text,
#   picked_title text,
#   picked_type text,
#   picked_stack text[],
#   difficulty text,
#   budget integer,
#   problem_type text,
#   solution_count integer,
#   created_at timestamptz DEFAULT now()
# );
# CREATE INDEX idx_flex_runs_fingerprint ON flex_runs(fingerprint);


def get_client() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[MEMORY] Supabase not configured — skipping")
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[MEMORY] Client error: {e}")
        return None


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
    """Save a completed project run to Supabase."""
    client = get_client()
    if not client:
        return False
    try:
        client.table("flex_runs").insert({
            "fingerprint": fingerprint,
            "problem": problem,
            "picked_title": picked_title,
            "picked_type": picked_type,
            "picked_stack": picked_stack,
            "difficulty": difficulty,
            "budget": budget,
            "problem_type": problem_type,
            "solution_count": solution_count,
        }).execute()
        print(f"[MEMORY] Saved run for {fingerprint}: {picked_title}")
        return True
    except Exception as e:
        print(f"[MEMORY] Save error: {e}")
        return False


async def get_user_history(fingerprint: str, limit: int = 3) -> list:
    """Fetch the last N runs for a user fingerprint."""
    client = get_client()
    if not client:
        return []
    try:
        response = client.table("flex_runs") \
            .select("*") \
            .eq("fingerprint", fingerprint) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return response.data or []
    except Exception as e:
        print(f"[MEMORY] Fetch error: {e}")
        return []


def build_memory_context(history: list) -> str:
    """
    Convert user history into a context string for the orchestrator.
    Returns empty string if no history.
    """
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
    """
    Derive preference summary from history for logging.
    """
    if not history:
        return {}

    types = [r.get("picked_type") for r in history if r.get("picked_type")]
    difficulties = [r.get("difficulty") for r in history if r.get("difficulty")]
    budgets = [r.get("budget") for r in history if r.get("budget")]
    stacks = []
    for r in history:
        stacks.extend(r.get("picked_stack") or [])

    from collections import Counter
    return {
        "preferred_type": Counter(types).most_common(1)[0][0] if types else None,
        "preferred_difficulty": Counter(difficulties).most_common(1)[0][0] if difficulties else None,
        "avg_budget": int(sum(budgets) / len(budgets)) if budgets else None,
        "top_stacks": [s for s, _ in Counter(stacks).most_common(3)],
        "run_count": len(history),
    }
