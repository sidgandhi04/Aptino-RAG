import time
from agents.state import AgentState


class CaseAnalysisAgent:
    def __init__(self):
        pass

    def invoke(self, state: AgentState) -> AgentState:
        t0 = time.time()
        case = state.case

        # --- Detect missing required fields ---
        missing_fields = []
        if not case.documents:
            missing_fields.append("documents")
        if not case.hospital.get("name"):
            missing_fields.append("hospital_name")
        # For domiciliary/day_care, doctor_certificate, procedure_record, or medical_records are acceptable clinical docs
        has_clinical_doc = any(d in case.documents for d in [
            "discharge_summary", "procedure_record", "doctor_certificate", "medical_records"
        ])
        if not has_clinical_doc:
            missing_fields.append("clinical_documentation")
        if "itemized_bill" not in case.documents:
            missing_fields.append("itemized_bill")

        # Diagnosis-specific required documents (e.g. Cancer requires histopathology report)
        diagnosis_lower = case.treatment.get("diagnosis", "").lower()
        if "cancer" in diagnosis_lower:
            has_path_doc = any(d in case.documents for d in ["histopathology_report", "pathology_report", "biopsy_report"])
            if not has_path_doc:
                missing_fields.append("histopathology_report")

        # --- Check for evidence_context gaps (ONLY when evidence_context is explicitly provided with unconfirmed mandatory criteria) ---
        ev_ctx = case.evidence_context
        has_evidence_context = ev_ctx is not None

        if has_evidence_context:
            if ev_ctx.get("hospital_registered") is None:
                missing_fields.append("hospital_registration_status")
            if ev_ctx.get("medical_necessity_confirmed") is None:
                missing_fields.append("medical_necessity_confirmation")
            if ev_ctx.get("hospital_minimum_criteria_documented") is False:
                missing_fields.append("hospital_minimum_criteria_documentation")

        # --- Robustness: note and ignore irrelevant attributes ---
        trace_msg = "Case Analysis Agent: Extracted claim facts and built investigation plan."
        if case.irrelevant_attributes:
            trace_msg += f" Ignored irrelevant attributes: {list(case.irrelevant_attributes.keys())}."
        if missing_fields:
            trace_msg += f" Missing fields: {missing_fields}."

        # --- Build DYNAMIC investigation plan ---
        investigation_plan = []

        # Always check basic waiting periods
        investigation_plan.append("Check initial 30-day waiting period for any illness.")

        # Specific disease waiting periods
        diagnosis = case.treatment.get("diagnosis", "").lower()
        if any(kw in diagnosis for kw in ["cataract", "hernia", "fistula", "tonsil", "sinus",
                                           "kidney", "stone", "gall bladder", "gout", "dialysis",
                                           "joint replacement", "hysterectomy"]):
            investigation_plan.append(f"Check 1-year or 2-year specific disease waiting period for: {case.treatment.get('diagnosis')}.")

        # Pre-existing disease
        if case.treatment.get("pre_existing"):
            investigation_plan.append("Check 48-month pre-existing disease waiting period.")

        # Exclusions
        if case.treatment.get("experimental"):
            investigation_plan.append("Check exclusion for unproven/experimental treatments.")
        if any(kw in diagnosis for kw in ["cosmetic", "plastic surgery", "aesthetic"]):
            investigation_plan.append("Check exclusion for cosmetic/plastic surgery.")

        # Treatment type specific
        if case.treatment.get("type") == "domiciliary":
            investigation_plan.append("Check domiciliary hospitalization conditions, limits, and 20% BSI sub-limit.")
        if case.treatment.get("type") == "day_care" or case.treatment.get("admission_hours", 99) < 24:
            investigation_plan.append("Check day care / less-than-24-hours treatment coverage conditions.")

        # Portability
        if case.prior_policy or case.prior_insurer_continuous_years > 0:
            investigation_plan.append("Check portability rules and waiting period waivers for continuous prior coverage with Indian insurer.")

        # Sub-limits (always relevant)
        investigation_plan.append("Check sub-limits: Room rent (1% of SI per day), Doctor fees (25% of SI), Medicines (40% of SI), Ambulance (Rs. 1000).")

        # Pre/post hospitalization
        investigation_plan.append("Check pre-hospitalization (30 days) and post-hospitalization (60 days) expense coverage windows.")

        # Hospital definition — only flag when evidence_context explicitly says unknown OR non-network + unknown
        if has_evidence_context and ev_ctx.get("hospital_registered") is None:
            investigation_plan.append("Check policy definition of Hospital and whether the facility meets the minimum criteria.")
        elif not case.hospital.get("network_provider") and not has_evidence_context:
            investigation_plan.append("Note: non-network hospital but no evidence_context provided about registration status.")

        # Evidence gaps
        if missing_fields:
            investigation_plan.append(f"Flag insufficient evidence for: {', '.join(missing_fields)}.")

        state.missing_fields = missing_fields
        state.investigation_plan = investigation_plan
        state.trace.append(trace_msg)
        state.timings["case_analysis_agent"] = round(time.time() - t0, 3)

        return state
