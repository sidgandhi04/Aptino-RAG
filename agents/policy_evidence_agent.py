import time
from agents.state import AgentState
from retrieval.hybrid_search import HybridSearcher


class PolicyEvidenceAgent:
    def __init__(self, searcher: HybridSearcher = None):
        self.searcher = searcher

    def invoke(self, state: AgentState) -> AgentState:
        t0 = time.time()

        if not self.searcher:
            state.trace.append("Policy Evidence Agent: No searcher configured — skipping retrieval.")
            state.timings["policy_evidence_agent"] = round(time.time() - t0, 3)
            return state

        case = state.case

        # --- Build queries from the investigation plan + case facts ---
        queries = []

        # Always include generic waiting period query
        queries.append("Initial 30 days waiting period for any disease")

        # Diagnosis-specific query
        diagnosis = case.treatment.get("diagnosis", "")
        if diagnosis:
            queries.append(f"Waiting period and coverage for {diagnosis}")

        # Sub-limits query
        queries.append("Sub-limits for room rent ICU doctor fees medicines ambulance")

        # Treatment-type queries
        if case.treatment.get("type") == "domiciliary":
            queries.append("Domiciliary hospitalization treatment conditions limits")
        if case.treatment.get("type") == "day_care" or case.treatment.get("admission_hours", 99) < 24:
            queries.append("Day care treatment less than 24 hours conditions")

        # Exclusions
        if case.treatment.get("experimental"):
            queries.append("Unproven experimental treatment exclusions")
        if "cosmetic" in diagnosis.lower():
            queries.append("Cosmetic plastic surgery exclusion")

        # Pre-existing disease
        if case.treatment.get("pre_existing"):
            queries.append("Pre-existing disease 48 month waiting period coverage")

        # Portability / prior coverage
        if case.prior_policy or case.prior_insurer_continuous_years > 0:
            queries.append("Portability prior insurer continuous coverage waiting period waiver")

        # Pre/post hospitalization
        queries.append("Pre-hospitalization 30 days post-hospitalization 60 days expenses")

        # Hospital definition (when evidence is uncertain)
        ev_ctx = case.evidence_context or {}
        if ev_ctx.get("hospital_registered") is None or not case.hospital.get("network_provider"):
            queries.append("Hospital definition minimum criteria registration")

        # --- Run queries through hybrid search ---
        all_evidence = {}
        for q in queries:
            try:
                results = self.searcher.search(q, top_k=3)
                for r in results:
                    chunk_id = r["chunk_id"]
                    if chunk_id not in all_evidence:
                        all_evidence[chunk_id] = r
            except Exception as e:
                print(f"Search failed for query '{q}': {e}")

        state.retrieved_evidence = list(all_evidence.values())
        state.trace.append(
            f"Policy Evidence Agent: Ran {len(queries)} queries, retrieved {len(state.retrieved_evidence)} distinct evidence chunks."
        )
        state.timings["policy_evidence_agent"] = round(time.time() - t0, 3)

        return state
