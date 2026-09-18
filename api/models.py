from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class ClaimRequest(BaseModel):
    case_id: str
    policy_id: str
    policy_start_date: str
    claim_date: str
    sum_insured_inr: float
    continuous_coverage_months: int
    prior_insurer_continuous_years: int = 0
    patient: Dict[str, Any]
    hospital: Dict[str, Any]
    treatment: Dict[str, Any]
    expenses_inr: Dict[str, float]
    documents: List[str]
    evidence_context: Optional[Dict[str, Any]] = None
    prior_policy: Optional[Dict[str, Any]] = None
    irrelevant_attributes: Optional[Dict[str, Any]] = None
    # Tolerate unknown fields from the test cases
    expense_timing: Optional[Dict[str, Any]] = None
    task: Optional[str] = None

class CitationModel(BaseModel):
    claim: str
    source: str
    page: int
    section: str
    chunk_id: str

class ValidationModel(BaseModel):
    status: Literal["PASS", "FAIL", "PENDING"]
    unsupported_claims: List[str]

class AnalyzeResponse(BaseModel):
    case_id: str
    decision: Literal["ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE", "NOT_ADMISSIBLE", "NEEDS_REVIEW", "PENDING"]
    confidence: float
    key_findings: List[str]
    applicable_limits: List[str]
    missing_evidence: List[str]
    citations: List[CitationModel]
    validation: ValidationModel
    trace: List[str]
    timings: Dict[str, float] = {}
