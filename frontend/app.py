import streamlit as st
import json
import os
import requests
import glob
import time
from dotenv import load_dotenv

load_dotenv()

# Constants
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Aptino Health Claims RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom CSS
css_path = "frontend/style.css"
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Aptino AI — Health Insurance Claims Decision Engine")


# ── Helpers ──────────────────────────────────────────────────
@st.cache_data
def load_cases():
    cases = []
    public_path = "Given Data/candidate_data/public_test_cases.json"
    if os.path.exists(public_path):
        with open(public_path, "r") as f:
            cases.extend(json.load(f))
    custom_paths = sorted(glob.glob("data/custom_cases/*.json"))
    for cp in custom_paths:
        with open(cp, "r") as f:
            cases.append(json.load(f))
    return {c["case_id"]: c for c in cases}


def _decision_color(decision: str) -> str:
    """Return an emoji + color hint for the decision status."""
    return {
        "ADMISSIBLE": "🟢",
        "ADMISSIBLE_WITH_LIMITS": "🟡",
        "PARTIALLY_ADMISSIBLE": "🟠",
        "NOT_ADMISSIBLE": "🔴",
        "NEEDS_REVIEW": "⚠️",
    }.get(decision, "⬜")


def _state_to_dict(raw_result):
    """Normalize Pydantic state objects into standard dictionaries for UI rendering."""
    if hasattr(raw_result, "model_dump"):
        result = raw_result.model_dump()
    elif isinstance(raw_result, dict):
        result = dict(raw_result)
    else:
        result = {}

    # Normalize citations list
    citations = []
    for c in result.get("citations", []):
        if hasattr(c, "model_dump"):
            citations.append(c.model_dump())
        elif isinstance(c, dict):
            citations.append(c)
        elif hasattr(c, "__dict__"):
            citations.append(c.__dict__)
    result["citations"] = citations

    # Normalize validation object
    val = result.get("validation", {})
    if hasattr(val, "model_dump"):
        result["validation"] = val.model_dump()
    elif hasattr(val, "__dict__"):
        result["validation"] = val.__dict__
    elif not isinstance(val, dict):
        result["validation"] = {
            "status": getattr(val, "status", "PASS"),
            "unsupported_claims": getattr(val, "unsupported_claims", [])
        }

    return result


# ── Sidebar: Case selection / upload / paste ─────────────────
st.sidebar.header("Claim Case Input")
input_mode = st.sidebar.radio(
    "How would you like to provide the case?",
    ["Select from library", "Upload JSON file", "Paste JSON"],
    index=0,
)

case_data = None

if input_mode == "Select from library":
    cases_dict = load_cases()
    if not cases_dict:
        st.error("No test cases found.")
        st.stop()
    selected_case_id = st.sidebar.selectbox("Case ID", list(cases_dict.keys()))
    case_data = cases_dict[selected_case_id]

elif input_mode == "Upload JSON file":
    uploaded = st.sidebar.file_uploader("Upload a claim case JSON", type=["json"])
    if uploaded is not None:
        try:
            case_data = json.load(uploaded)
            if "case_id" not in case_data:
                st.sidebar.error("JSON must contain a 'case_id' field.")
                case_data = None
        except json.JSONDecodeError as e:
            st.sidebar.error(f"Invalid JSON: {e}")

elif input_mode == "Paste JSON":
    raw_json = st.sidebar.text_area(
        "Paste claim case JSON here",
        height=300,
        placeholder='{\n  "case_id": "MY-001",\n  "policy_id": "...",\n  ...\n}',
    )
    if raw_json.strip():
        try:
            case_data = json.loads(raw_json)
            if "case_id" not in case_data:
                st.sidebar.error("JSON must contain a 'case_id' field.")
                case_data = None
        except json.JSONDecodeError as e:
            st.sidebar.error(f"Invalid JSON: {e}")

if case_data is None:
    st.info("Select, upload, or paste a claim case to begin.")
    st.stop()

# ── Main content ─────────────────────────────────────────────
st.header(f"Case: {case_data.get('case_id', 'Unknown')}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Claim Facts")
    st.json(case_data)

with col2:
    st.subheader("Analysis & Decision")

    if st.button("🔍 Run AI Analysis", use_container_width=True):
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Initializing RAG pipeline…")

        try:
            # 1. Start async task
            response = requests.post(f"{API_URL}/analyze_async", json=case_data, timeout=30)
            if response.status_code != 200:
                st.error(f"Error {response.status_code}: {response.text}")
                st.stop()

            task_id = response.json()["task_id"]

            # 2. Poll for status
            result = None
            while True:
                time.sleep(1.5)
                status_res = requests.get(f"{API_URL}/task_status/{task_id}", timeout=10).json()

                if status_res["status"] == "done":
                    result = status_res["result"]
                    break
                elif status_res["status"] == "error":
                    st.error(f"❌ Backend error: {status_res['result']}")
                    st.stop()

                msgs = status_res.get("messages", [])
                if msgs:
                    latest = msgs[-1]
                    if "Rate Limit" in latest:
                        status_placeholder.warning(f"⏳ {latest}")
                    else:
                        status_placeholder.info(f"⏳ {latest}")
                else:
                    status_placeholder.info("⏳ Analyzing claim via RAG pipeline…")

            status_placeholder.empty()

            # ── Decision display ────────────────────────
            decision = result.get("decision", "PENDING")
            confidence = result.get("confidence", 0.0)
            icon = _decision_color(decision)

            # Abstention banner
            if decision == "NEEDS_REVIEW":
                st.warning(
                    "⚠️ **System Abstained — Insufficient Evidence**\n\n"
                    "The system could not reach a confident decision because required "
                    "evidence or policy support is missing. A human reviewer should "
                    "investigate the items listed under *Missing Evidence* below.",
                    icon="⚠️",
                )

            st.metric(
                label="Decision",
                value=f"{icon} {decision}",
                delta=f"Confidence: {confidence:.0%}",
            )

            # Key Findings
            st.markdown("### 📋 Key Findings")
            findings = result.get("key_findings", [])
            if findings:
                for f in findings:
                    st.markdown(f"- {f}")
            else:
                st.caption("No key findings reported.")

            # Applicable Limits
            st.markdown("### 📏 Applicable Limits")
            limits = result.get("applicable_limits", [])
            if limits:
                for lim in limits:
                    st.markdown(f"- {lim}")
            else:
                st.caption("No applicable limits identified.")

            # Missing Evidence (always shown)
            st.markdown("### 🔍 Missing Evidence")
            missing = result.get("missing_evidence", [])
            if missing:
                for me in missing:
                    st.warning(me)
            else:
                st.success("No missing evidence — all required information is present.")

            # Citations
            st.markdown("### 📖 Policy Citations")
            citations = result.get("citations", [])
            if citations:
                for cit in citations:
                    with st.expander(
                        f"Page {cit.get('page', '?')} · {cit.get('section', 'Unknown section')}"
                    ):
                        st.markdown(f"**Claim:** {cit['claim']}")
                        st.markdown(f"**Source:** {cit.get('source', 'policy.pdf')}")
                        st.caption(f"Chunk ID: {cit.get('chunk_id', '-')}")
            else:
                st.caption("No citations provided.")

            # Validation
            st.markdown("### ✅ Validation Critic")
            val = result.get("validation", {})
            val_status = val.get("status", "PENDING")
            if val_status == "PASS":
                st.success(f"Validation: **{val_status}** — all claims are evidence-grounded.")
            else:
                st.error(f"Validation: **{val_status}**")
                for uc in val.get("unsupported_claims", []):
                    st.error(f"  ↳ {uc}")

            # Execution Trace with timing
            st.markdown("### 🕐 Execution Trace")
            timings = result.get("timings", {})
            trace = result.get("trace", [])

            if trace:
                trace_rows = []
                for entry in trace:
                    # Parse agent name from trace string  (format: "Agent Name: action")
                    parts = entry.split(":", 1)
                    agent_name = parts[0].strip() if len(parts) > 1 else "System"
                    action = parts[1].strip() if len(parts) > 1 else entry
                    # Find timing for this agent
                    agent_key = agent_name.lower().replace(" ", "_")
                    elapsed = timings.get(agent_key, None)
                    elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "—"
                    trace_rows.append({
                        "Agent": agent_name,
                        "Action": action,
                        "Duration": elapsed_str,
                    })

                st.table(trace_rows)

                total_time = sum(timings.values()) if timings else 0
                if total_time > 0:
                    st.caption(f"Total pipeline time: **{total_time:.2f}s**")
            else:
                st.caption("No trace data available.")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            status_placeholder.info("⏳ Executing RAG workflow in-process…")
            try:
                from agents.graph import create_graph
                from agents.state import AgentState, ClaimCase
                workflow = create_graph()
                initial_state = AgentState(case=ClaimCase(**case_data))
                raw_result = workflow.invoke(initial_state)
                result = _state_to_dict(raw_result)
            except Exception as graph_err:
                st.error(f"❌ Workflow execution error: {graph_err}")
                st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()
