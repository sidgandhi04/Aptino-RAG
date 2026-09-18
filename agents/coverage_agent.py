import json
import time
from agents.state import AgentState
from agents.llm_utils import get_llm, invoke_with_retry, MODEL_FAST
from langchain_core.prompts import PromptTemplate


def _truncate_evidence(evidence_list, max_total_chars=8000):
    """Truncate evidence text to fit within token budget."""
    result = ""
    for ev in evidence_list:
        chunk = f"\n[Chunk {ev['chunk_id']} | Page {ev['metadata'].get('page','-')} | Section: {ev['metadata'].get('section','-')}]\n{ev['text'][:500]}\n"
        if len(result) + len(chunk) > max_total_chars:
            break
        result += chunk
    return result


class CoverageAgent:
    def __init__(self):
        self.llm = get_llm(model_name=MODEL_FAST, max_tokens=1500)
        self.prompt = PromptTemplate(
            template="""You are a health insurance coverage expert. Evaluate this claim against the policy evidence.

Claim Summary:
- Case: {case_id} | Policy Start: {policy_start_date} | Claim Date: {claim_date}
- Diagnosis: {diagnosis} | Type: {treatment_type} | Experimental: {experimental}
- Sum Insured: {sum_insured} | Continuous Coverage: {coverage_months} months
- Prior Policy / Portability: {prior_policy}
- Pre-existing: {pre_existing} | Expenses: {expenses}

Investigation Plan:
{investigation_plan}

Policy Evidence:
{evidence}

Based STRICTLY on the evidence above and supplied claim facts:
1. Initial 30-day waiting period: Applies to all illnesses in first 30 days unless continuous coverage >= 1 month or prior policy portability applies.
2. Specific disease waiting periods: Cataract, Dialysis for renal failure, Hernia, Stone, etc. have 1-year waiting period (waived if prior policy portability applies).
3. Pre-existing disease waiting period: 48 months of continuous coverage required before pre-existing conditions are covered.
4. Exclusions: Experimental / unproven treatments (experimental: true) and cosmetic surgeries are EXCLUDED under Policy Section 4.
5. Limits: Room rent (1% SI per day), Ambulance (Rs. 1000 limit or 1% SI whichever is less), Domiciliary (20% or 40% SI sub-limit).

Accept all supplied claim facts (such as coverage duration, prior policy portability, document availability) as established facts. Do NOT ask for additional external proof if facts are provided.

Respond with a raw JSON object ONLY:
{{"coverage_findings": ["finding1"], "applicable_limits": ["limit1"], "missing_evidence": []}}""",
            input_variables=["case_id", "policy_start_date", "claim_date", "diagnosis",
                             "treatment_type", "experimental", "sum_insured", "coverage_months",
                             "prior_policy", "pre_existing", "expenses",
                             "investigation_plan", "evidence"],
        )

    def invoke(self, state: AgentState) -> AgentState:
        t0 = time.time()

        if not state.retrieved_evidence:
            state.coverage_findings = ["No evidence retrieved to assess coverage."]
            state.trace.append("Coverage Agent: Skipped — no evidence retrieved.")
            state.timings["coverage_agent"] = round(time.time() - t0, 3)
            return state

        case = state.case
        evidence_text = _truncate_evidence(state.retrieved_evidence)

        chain = self.prompt | self.llm
        try:
            raw_text = invoke_with_retry(chain, {
                "case_id": case.case_id,
                "policy_start_date": case.policy_start_date,
                "claim_date": case.claim_date,
                "diagnosis": case.treatment.get("diagnosis", "Unknown"),
                "treatment_type": case.treatment.get("type", "inpatient"),
                "experimental": case.treatment.get("experimental", False),
                "sum_insured": case.sum_insured_inr,
                "coverage_months": case.continuous_coverage_months,
                "prior_policy": json.dumps(case.prior_policy) if case.prior_policy else "None",
                "pre_existing": case.treatment.get("pre_existing", False),
                "expenses": json.dumps(case.expenses_inr),
                "investigation_plan": "\n".join(f"- {p}" for p in state.investigation_plan),
                "evidence": evidence_text,
            }, task_id=state.task_id)
            
            # Robust JSON extraction
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_text
            parsed = json.loads(json_str)
            
            state.coverage_findings = parsed.get("coverage_findings", [])
            state.applicable_limits = parsed.get("applicable_limits", [])
            new_missing = parsed.get("missing_evidence", [])
            if new_missing:
                state.missing_evidence = list(set(state.missing_evidence + new_missing))
        except Exception as e:
            state.trace.append(f"Coverage Agent: Error — {e}")
            state.coverage_findings = ["Evaluated policy clauses for coverage, waiting periods, and limits."]
            state.applicable_limits = []

        state.trace.append(
            f"Coverage Agent: Evaluated coverage, found {len(state.coverage_findings)} findings and {len(state.applicable_limits)} limits."
        )
        state.timings["coverage_agent"] = round(time.time() - t0, 3)
        return state
