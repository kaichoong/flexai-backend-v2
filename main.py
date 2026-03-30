"""
Flex AI — FastAPI backend
Exposes the multi-agent graph as a REST API with SSE streaming
so the frontend can show real-time agent progress.
"""

import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from graph import flex_graph

load_dotenv()

app = FastAPI(title="Flex AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectRequest(BaseModel):
    problem: str
    budget: int = 50
    fingerprint: str = ""


class SaveRunRequest(BaseModel):
    fingerprint: str
    problem: str
    picked_title: str
    picked_type: str
    picked_stack: list[str] = []
    difficulty: str = "intermediate"
    budget: int = 50
    problem_type: str = "software"
    solution_count: int = 3


class VideoScriptRequest(BaseModel):
    project_title: str
    project_type: str
    stack: list[str]
    problem_scope: str
    step_title: str
    step_desc: str
    step_type: str
    difficulty: str


@app.get("/")
async def root():
    return {"status": "Flex AI backend running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/projects/stream")
async def stream_projects(request: ProjectRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    async def event_stream():
        try:
            initial_state = {
                "problem": request.problem,
                "budget": request.budget,
                "fingerprint": request.fingerprint,
                "orchestrator": None,
                "planner": None,
                "stack_scout": None,
                "budget_bot": None,
                "tutorial": None,
                "code_agent": None,
                "tools_sourcer": None,
                "critic": None,
                "video_agent": None,
                "log": [],
                "error": None,
            }

            yield f"data: {json.dumps({'type': 'log', 'message': 'Connected — Orchestrator + 7 agents starting…'})}\n\n"
            await asyncio.sleep(0.1)

            last_ping = asyncio.get_event_loop().time()

            async for step in flex_graph.astream(initial_state):
                node_name = list(step.keys())[0] if step else None
                node_state = step.get(node_name, {}) if node_name else {}

                now = asyncio.get_event_loop().time()
                if now - last_ping > 10:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    last_ping = now

                for entry in node_state.get("log", []):
                    yield f"data: {json.dumps({'type': 'log', 'message': entry})}\n\n"
                    await asyncio.sleep(0.05)

                if node_name == "synthesise" and "projects" in node_state:
                    yield f"data: {json.dumps({'type': 'result', 'projects': node_state['projects']})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/api/projects")
async def get_projects(request: ProjectRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        initial_state = {
            "problem": request.problem,
            "budget": request.budget,
            "fingerprint": request.fingerprint,
            "orchestrator": None,
            "planner": None,
            "stack_scout": None,
            "budget_bot": None,
            "tutorial": None,
            "code_agent": None,
            "tools_sourcer": None,
            "critic": None,
            "video_agent": None,
            "log": [],
            "error": None,
        }

        final_state = await flex_graph.ainvoke(initial_state)

        return {
            "projects": final_state.get("projects", []),
            "log": final_state.get("log", []),
            "planner": final_state.get("planner", {}),
            "orchestrator": final_state.get("orchestrator", {}),
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR /api/projects: {tb}")
        raise HTTPException(status_code=500, detail=f"{str(e)} | Traceback: {tb[-800:]}")


@app.post("/api/video/script")
async def generate_video_script(request: VideoScriptRequest):
    from agents import call_gemini, parse_json

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        system = """You are the Video Agent in Flex AI. Write a 60-second tutorial video script specific to the project and step.
Respond ONLY with valid JSON, no markdown:
{
  "title": "short video title max 6 words",
  "narration": "spoken script 80-100 words, reference exact project name and tools, friendly and encouraging",
  "captions": [{"t": 0, "text": "caption"}, {"t": 6, "text": "next"}],
  "code_lines": ["line1", "line2", "line3", "line4", "line5"]
}
Include 8-10 caption objects. code_lines: key real code lines max 8."""

        user = f"""Problem: {request.problem_scope}
Project: {request.project_title} ({request.project_type})
Stack: {', '.join(request.stack)}
Difficulty: {request.difficulty}
Step type: {request.step_type}
Step: "{request.step_title}"
Description: {request.step_desc}

Write the 60-second video script."""

        result = await call_gemini(system, user, 800)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tutorial/steps")
async def generate_tutorial_steps(request: dict):
    from agents import call_gemini

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        project = request.get("project", {})
        problem = request.get("problem", "")
        starter_code = project.get("starter_code") or {}
        tools = project.get("tools") or []
        tools_str = ", ".join([t.get("name","") + " (" + t.get("url","") + ")" for t in tools])

        system = """You are the Tutorial Agent in Flex AI generating detailed step-by-step instructions.
Respond ONLY with valid JSON, no markdown:
{
  "steps": [
    {
      "id": 0,
      "type": "concept | setup | hardware | code | terminal | config | deploy | test",
      "title": "...",
      "desc": "...",
      "dur": "X min",
      "content": [
        {"t": "prose", "v": "HTML with <strong> tags ok"},
        {"t": "callout", "v": "info | warn | success", "text": "..."},
        {"t": "code", "lang": "python | bash | javascript", "editable": true, "code": "REAL code", "expected": "expected output"},
        {"t": "wire", "label": "component wiring"}
      ],
      "errors": ["real error 1", "real error 2"],
      "verify": "what user should see to confirm step worked"
    }
  ]
}
Rules: 6-8 steps, real runnable code only, wire type only on hardware steps."""

        user = f"""Problem: {problem}
Project: {project.get('title','')} ({project.get('type','')})
Stack: {', '.join(project.get('stack',[]))}
Difficulty: {project.get('difficulty','')}
Description: {project.get('description','')}

Starter code ({starter_code.get('filename','main.py')}):
Install: {starter_code.get('install','')}
{starter_code.get('code','')}

Tools: {tools_str}

Write the complete tutorial with real code."""

        result = await call_gemini(system, user, 4000)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug")
async def debug():
    from agents import call_gemini
    result = await call_gemini(
        "Respond ONLY with valid JSON, no markdown: {\"status\": \"ok\", \"msg\": \"groq working\"}",
        "ping",
        100
    )
    return {"groq_response": result, "groq_key_set": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/api/debug/full")
async def debug_full():
    from agents import planner_agent, stack_scout_agent
    state = {"problem": "I want to build a simple todo app", "budget": 50, "log": [], "error": None}
    s1 = await planner_agent(state)
    s2 = await stack_scout_agent(s1)
    return {"planner": s1.get("planner"), "stack_scout": s2.get("stack_scout"), "log": s2.get("log")}


@app.post("/api/memory/save")
async def memory_save(request: SaveRunRequest):
    from memory import save_run
    success = await save_run(
        fingerprint=request.fingerprint,
        problem=request.problem,
        picked_title=request.picked_title,
        picked_type=request.picked_type,
        picked_stack=request.picked_stack,
        difficulty=request.difficulty,
        budget=request.budget,
        problem_type=request.problem_type,
        solution_count=request.solution_count,
    )
    return {"saved": success}


@app.get("/api/memory/{fingerprint}")
async def memory_get(fingerprint: str):
    from memory import get_user_history, get_user_preferences
    history = await get_user_history(fingerprint)
    prefs = get_user_preferences(history)
    return {"history": history, "preferences": prefs}


@app.post("/api/audio/generate")
async def generate_audio(request: dict):
    """
    Phase 5: ElevenLabs TTS — generate real voiceover audio for a video script.
    Returns base64-encoded MP3 audio.
    """
    import base64

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")

    narration = request.get("narration", "")
    if not narration:
        raise HTTPException(status_code=400, detail="narration is required")

    # Use Rachel voice — clear, friendly, professional
    voice_id = request.get("voice_id", "21m00Tcm4TlvDq8ikWAM")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": narration,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.3,
                        "use_speaker_boost": True
                    }
                }
            )

        if response.status_code != 200:
            print(f"[ELEVENLABS] Error: {response.status_code} {response.text[:200]}")
            raise HTTPException(status_code=500, detail=f"ElevenLabs error: {response.status_code}")

        audio_b64 = base64.b64encode(response.content).decode("utf-8")
        print(f"[ELEVENLABS] Generated {len(response.content)} bytes of audio")
        return {"audio_b64": audio_b64, "format": "mp3"}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ElevenLabs request timed out")
    except Exception as e:
        print(f"[ELEVENLABS] Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: dict):
    from agents import call_gemini_text

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        system = request.get("system", "You are a helpful AI assistant. Be concise.")
        user = request.get("user", "")
        max_tokens = request.get("max_tokens", 300)
        text = await call_gemini_text(system, user, max_tokens)
        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
