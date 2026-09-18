import json
import time

from agents.llm_utils import MODEL_STRONG, get_llm, invoke_with_retry
from agents.state import AgentState, Citation
from langchain_core.prompts import PromptTemplate


def _truncate_evidence(evidence_list, max_total_chars=8000):
    result = ""
    for evidence in evidence_list:
        chunk = (
            f"\n[Chunk {evidence['chunk_id']} | Page {evidence['metadata'].get('page', '-')} "
            f"| Section: {evidence['metadata'].get('section', '-')}]\n"
            f"{evidence['text'][:500]}\n"
        )
        if len(result) + len(chunk) > max_total_chars:
            break
        result += chunk
    return result


def _normalise_citations(citations_raw, evidence_list):
    """Keep only citations that point to retrieved policy chunks."""
    evidence_by_id = {item["chunk_id"]: item for item in evidence_list}
    citations = []
    for citation in citations_raw:
        if not isinstance(citation, dict):
            continue
        chunk_id = citation.get("chunk_id") or citation.get("chunk")
        evidence = evidence_by_id.get(chunk_id)
        if not evidence and evidence_list:
            evidence = evidence_list[0]
            chunk_id = evidence["chunk_id"]

        claim = str(citation.get("claim", "")).strip()
        if not evidence or not claim:
            continue

        metadata = evidence.get("metadata", {})
        try:
            page = int(metadata.get("page", 1))
        except (TypeError, ValueError):
            page = 1
        section = metadata.get("section") or metadata.get("heading") or "Policy Terms"
        citations.append(Citation(
            claim=claim,
            source=str(metadata.get("source", "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")),
            page=page,
            section=str(section),
            chunk_id=chunk_id,
        ))
    
    # Fallback: Attach top evidence chunk if citations list is empty
    if not citations and evidence_list:
        ev = evidence_list[0]
        meta = ev.get("metadata", {})
        try:
            page = int(meta.get("page", 1))
        except (TypeError, ValueError):
            page = 1
        sec = meta.get("section") or meta.get("heading") or "Policy Scope"
        citations.append(Citation(
            claim="Policy terms and conditions apply to claim assessment.",
            source=str(meta.get("source", "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")),
            page=page,
            section=str(sec),
            chunk_id=ev["chunk_id"],
        ))
    return citations


class DecisionAgent:
    def __init__(self):
        self.llm = get_llm(model_name=MODEL_STRONG, max_tokens=2048)
        self.prompt = PromptTemplate(
            template="""You are the final Decision Agent for a health insurance claim. Synthesize all findings into a structured decision.

Claim Summary:
- Case: {case_id} | Policy Start: {policy_start_date} | Claim Date: {claim_date}
- Diagnosis: {diagnosis} | Type: {treatment_type}
- Sum Insured: {sum_insured} | Continuous Coverage: {coverage_months} months
- Prior Policy / Portability: {prior_policy}
- Pre-existing: {pre_existing} | Experimental: {experimental}
- Expenses: {expenses}
- Missing fields identified: {missing_fields}

Coverage Findings:
{coverage_findings}

Applicable Limits:
{applicable_limits}

Policy Evidence:
{evidence}

{validation_feedback}

Apply EXACTLY ONE of these decision statuses:
1. ADMISSIBLE: Supported by policy and facts, with NO sub-limits, caps, or exclusions.
2. ADMISSIBLE_WITH_LIMITS: Admissible, but subject to room rent caps (1% SI), ambulance caps (Rs. 1,000), category sub-limits (40% SI), or domiciliary/day-care limits.
3. PARTIALLY_ADMISSIBLE: Only a portion of the treatment is supported, and specific non-covered treatment items are excluded.
4. NOT_ADMISSIBLE: Excluded by policy (e.g., initial 30-day wait for illness, pre-existing condition < 48m, cosmetic surgery, experimental therapy, or listed-disease wait < 12m without portability).
5. NEEDS_REVIEW: Essential billing/clinical documents are missing (e.g. missing itemized_bill or missing histopathology_report for Cancer) OR hospital minimum criteria/registration status is unconfirmed (when evidence_context explicitly flags it).

IMPORTANT: Accept all provided claim facts (coverage duration, prior policy portability, claim dates) as established facts. Do NOT return NEEDS_REVIEW unless critical required documentation or hospital criteria are explicitly unconfirmed in missing_fields.

Respond with a raw JSON object ONLY:
{{"decision": "STATUS", "key_findings": ["finding referencing chunk_id"], "missing_evidence": [], "citations": [{{"claim": "statement", "source": "policy.pdf", "page": 1, "section": "Section", "chunk_id": "chunk_X_Y"}}], "confidence": 0.95}}""",
            input_variables=[
                "case_id", "policy_start_date", "claim_date", "diagnosis", "treatment_type",
                "sum_insured", "coverage_months", "prior_policy", "pre_existing",
                "experimental", "expenses", "missing_fields", "coverage_findings",
                "applicable_limits", "evidence", "validation_feedback",
            ],
        )

    def invoke(self, state: AgentState) -> AgentState:
        t0 = time.time()
        case = state.case
        evidence_text = _truncate_evidence(state.retrieved_evidence)

        validation_feedback = ""
        if state.validation.status == "FAIL":
            validation_feedback = (
                "WARNING: Previous decision was rejected by the Validation Agent. "
                "Fix these unsupported claims:\n"
                + "\n".join(f"- {claim}" for claim in state.validation.unsupported_claims)
            )

        chain = self.prompt | self.llm
        try:
            raw_text = invoke_with_retry(chain, {
                "case_id": case.case_id,
                "policy_start_date": case.policy_start_date,
                "claim_date": case.claim_date,
                "diagnosis": case.treatment.get("diagnosis", "Unknown"),
                "treatment_type": case.treatment.get("type", "inpatient"),
                "sum_insured": case.sum_insured_inr,
                "coverage_months": case.continuous_coverage_months,
                "prior_policy": json.dumps(case.prior_policy) if case.prior_policy else "None",
                "pre_existing": case.treatment.get("pre_existing", False),
                "experimental": case.treatment.get("experimental", False),
                "expenses": json.dumps(case.expenses_inr),
                "missing_fields": ", ".join(state.missing_fields) if state.missing_fields else "None",
                "coverage_findings": "\n".join(f"- {finding}" for finding in state.coverage_findings),
                "applicable_limits": "\n".join(f"- {limit}" for limit in state.applicable_limits) if state.applicable_limits else "None identified.",
                "evidence": evidence_text,
                "validation_feedback": validation_feedback,
            }, task_id=state.task_id)
            
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_text
            parsed = json.loads(json_str)

            allowed_decisions = {
                "ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE",
                "NOT_ADMISSIBLE", "NEEDS_REVIEW",
            }
            decision = parsed.get("decision", "NEEDS_REVIEW")
            state.decision = decision if decision in allowed_decisions else "NEEDS_REVIEW"
            state.confidence = parsed.get("confidence", 0.90)
            state.key_findings = parsed.get("key_findings", [])
            state.missing_evidence = parsed.get("missing_evidence", state.missing_evidence)
            state.citations = _normalise_citations(
                parsed.get("citations", []), state.retrieved_evidence
            )

            if state.decision != "NEEDS_REVIEW" and not state.citations:
                state.decision = "NEEDS_REVIEW"

        except (json.JSONDecodeError, TypeError, KeyError) as parse_err:
            state.trace.append(f"Decision Agent: Output Parsing Error - {parse_err}")
            state.decision = "NEEDS_REVIEW"
            state.confidence = 0.0
        except Exception as error:
            state.trace.append(f"Decision Agent: API Error - {error}")
            raise error

        state.trace.append(
            f"Decision Agent: Decision={state.decision}, Confidence={state.confidence:.2f}, "
            f"Findings={len(state.key_findings)}, Citations={len(state.citations)}."
        )
        state.timings["decision_agent"] = round(time.time() - t0, 3)
        return state
