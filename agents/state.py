from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import operator
from typing import Annotated

class ClaimCase(BaseModel):
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
    # Extra fields from test cases (tolerate gracefully)
    expense_timing: Optional[Dict[str, Any]] = None
    task: Optional[str] = None

class Citation(BaseModel):
    claim: str
    source: str
    page: int
    section: str
    chunk_id: str

class ValidationResult(BaseModel):
    status: Literal["PASS", "FAIL", "PENDING"] = "PENDING"
    unsupported_claims: List[str] = []

class AgentState(BaseModel):
    case: ClaimCase
    task_id: str = ""
    investigation_plan: List[str] = []
    missing_fields: List[str] = []
    
    # Evidence Agent output
    retrieved_evidence: List[Dict[str, Any]] = []
    
    # Coverage Agent output
    coverage_findings: List[str] = []
    applicable_limits: List[str] = []
    
    # Decision Agent output
    decision: Literal["ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE", "NOT_ADMISSIBLE", "NEEDS_REVIEW", "PENDING"] = "PENDING"
    confidence: float = 0.0
    key_findings: List[str] = []
    missing_evidence: List[str] = []
    citations: List[Citation] = []
    
    # Validation node output
    validation: ValidationResult = Field(default_factory=ValidationResult)
    validation_attempts: int = 0
    
    # Trace
    trace: Annotated[List[str], operator.add] = []
    
    # Per-agent timing (seconds)
    timings: Dict[str, float] = {}
