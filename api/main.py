import uuid
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.models import ClaimRequest, AnalyzeResponse
from api.tasks import TASKS
from agents.graph import create_graph
from agents.state import AgentState, ClaimCase

app = FastAPI(title="Aptino Health Insurance Claims RAG API")

# Initialize LangGraph workflow
try:
    workflow = create_graph()
except Exception as e:
    print(f"Error initializing workflow: {e}")
    workflow = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request schema", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "workflow_initialized": workflow is not None}


def _build_response(final_state, case_id: str) -> dict:
    """Build an AnalyzeResponse dict from the final LangGraph state."""
    return AnalyzeResponse(
        case_id=case_id,
        decision=final_state["decision"],
        confidence=final_state["confidence"],
        key_findings=final_state["key_findings"],
        applicable_limits=final_state["applicable_limits"],
        missing_evidence=final_state["missing_evidence"],
        citations=[c.model_dump() for c in final_state["citations"]],
        validation=final_state["validation"].model_dump(),
        trace=final_state["trace"],
        timings=final_state.get("timings", {}),
    ).model_dump()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_claim(request: ClaimRequest):
    if not workflow:
        raise HTTPException(status_code=500, detail="Workflow not initialized.")

    initial_state = AgentState(case=ClaimCase(**request.model_dump(exclude_none=True)))
    final_state = workflow.invoke(initial_state)
    resp = _build_response(final_state, request.case_id)
    return resp


@app.post("/analyze_async")
def analyze_async(request: ClaimRequest, background_tasks: BackgroundTasks):
    if not workflow:
        raise HTTPException(status_code=500, detail="Workflow not initialized.")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "running", "messages": [], "result": None}

    def run_graph(t_id, req_data, case_id):
        try:
            initial_state = AgentState(
                case=ClaimCase(**req_data), task_id=t_id
            )
            final_state = workflow.invoke(initial_state)
            result = _build_response(final_state, case_id)
            TASKS[t_id]["status"] = "done"
            TASKS[t_id]["result"] = result
        except Exception as e:
            TASKS[t_id]["status"] = "error"
            TASKS[t_id]["result"] = str(e)

    background_tasks.add_task(
        run_graph, task_id, request.model_dump(exclude_none=True), request.case_id
    )
    return {"task_id": task_id}


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS[task_id]
