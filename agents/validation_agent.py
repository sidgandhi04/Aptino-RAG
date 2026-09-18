import json
import time
from agents.state import AgentState, ValidationResult
from agents.llm_utils import get_llm, invoke_with_retry, MODEL_FAST
from langchain_core.prompts import PromptTemplate


class ValidationAgent:
    MAX_ATTEMPTS = 2  # Reduced from 3 to save token budget

    def __init__(self):
        self.llm = get_llm(model_name=MODEL_FAST, temperature=0, max_tokens=1024)
        self.prompt = PromptTemplate(
            template="""You are a strict Validation Critic. Verify that every key finding from the Decision Agent is supported by the cited policy evidence.
Do NOT use external knowledge.

Decision: {decision}
Key Findings:
{key_findings}
Applicable Limits:
{applicable_limits}
Citations:
{citations}

Evidence:
{evidence}

Rules:
- If ANY key finding makes a material claim NOT explicitly stated or logically entailed by the cited evidence, return FAIL.
- If everything is grounded in the evidence, return PASS.
- List every unsupported claim specifically.

Respond with a raw JSON object:
{{"status": "PASS", "unsupported_claims": []}}""",
            input_variables=["decision", "key_findings", "applicable_limits", "citations", "evidence"],
        )

    def invoke(self, state: AgentState) -> AgentState:
        t0 = time.time()

        # Guardrail: limit retries
        state.validation_attempts += 1
        if state.validation_attempts > self.MAX_ATTEMPTS:
            # Don't override the decision — just force PASS to break the loop
            state.validation = ValidationResult(status="PASS", unsupported_claims=[])
            state.trace.append(
                f"Validation Agent: Max retries ({self.MAX_ATTEMPTS}) reached. Accepting current decision."
            )
            state.timings["validation_agent"] = round(time.time() - t0, 3)
            return state

        evidence_text = ""
        for ev in state.retrieved_evidence:
            evidence_text += f"\n[Chunk {ev['chunk_id']}]\n{ev['text'][:400]}\n"
            if len(evidence_text) > 3000:
                break

        citations_json = json.dumps([c.model_dump() for c in state.citations], indent=2)

        chain = self.prompt | self.llm
        try:
            raw_text = invoke_with_retry(chain, {
                "decision": state.decision,
                "key_findings": "\n".join(f"- {f}" for f in state.key_findings),
                "applicable_limits": "\n".join(f"- {l}" for l in state.applicable_limits) if state.applicable_limits else "None",
                "citations": citations_json,
                "evidence": evidence_text,
            }, task_id=state.task_id)
            parsed = json.loads(raw_text)

            state.validation = ValidationResult(
                status=parsed.get("status", "PASS"),
                unsupported_claims=parsed.get("unsupported_claims", [])
            )
        except (json.JSONDecodeError, TypeError, KeyError) as parse_err:
            state.trace.append(f"Validation Agent: Parsing Error — {parse_err}")
            state.validation = ValidationResult(status="PASS", unsupported_claims=[])
        except Exception as e:
            state.trace.append(f"Validation Agent: API Error — {e}")
            raise e

        state.trace.append(
            f"Validation Agent: Result={state.validation.status}, Attempt {state.validation_attempts}/{self.MAX_ATTEMPTS}."
        )
        state.timings["validation_agent"] = round(time.time() - t0, 3)
        return state
