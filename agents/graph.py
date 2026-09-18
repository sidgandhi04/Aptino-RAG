from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.case_analysis_agent import CaseAnalysisAgent
from agents.policy_evidence_agent import PolicyEvidenceAgent
from agents.coverage_agent import CoverageAgent
from agents.decision_agent import DecisionAgent
from agents.validation_agent import ValidationAgent
from retrieval.hybrid_search import HybridSearcher

def create_graph(db_dir="chroma_db"):
    # Initialize shared resources
    try:
        searcher = HybridSearcher(db_dir=db_dir)
    except Exception as e:
        print(f"Warning: Failed to initialize HybridSearcher: {e}")
        searcher = None
        
    analysis_agent = CaseAnalysisAgent()
    evidence_agent = PolicyEvidenceAgent(searcher=searcher)
    coverage_agent = CoverageAgent()
    decision_agent = DecisionAgent()
    validation_agent = ValidationAgent()
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_case", analysis_agent.invoke)
    workflow.add_node("retrieve_evidence", evidence_agent.invoke)
    workflow.add_node("assess_coverage", coverage_agent.invoke)
    workflow.add_node("make_decision", decision_agent.invoke)
    workflow.add_node("validate_decision", validation_agent.invoke)
    
    # Define edges
    workflow.set_entry_point("analyze_case")
    workflow.add_edge("analyze_case", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "assess_coverage")
    workflow.add_edge("assess_coverage", "make_decision")
    workflow.add_edge("make_decision", "validate_decision")
    
    # Conditional edge for validation retry loop
    def route_validation(state: AgentState):
        if state.validation.status == "PASS":
            return END
        else:
            return "make_decision" # Loop back to decision to revise based on feedback
            
    workflow.add_conditional_edges(
        "validate_decision",
        route_validation,
        {
            END: END,
            "make_decision": "make_decision"
        }
    )
    
    return workflow.compile()
