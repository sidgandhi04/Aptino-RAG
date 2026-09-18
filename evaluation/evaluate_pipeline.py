import json
import os
import glob
import time
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import create_graph
from agents.state import AgentState, ClaimCase


def load_expected_outcomes():
    """Load expected outcomes for evaluation."""
    outcomes = {}
    path = "data/expected_outcomes.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            for d in data:
                outcomes[d["case_id"]] = d
    return outcomes


def evaluate():
    """Run end-to-end evaluation across all public + custom cases."""
    expected_outcomes = load_expected_outcomes()

    # Load all cases
    cases = []
    public_path = "Given Data/candidate_data/public_test_cases.json"
    if os.path.exists(public_path):
        with open(public_path, "r") as f:
            cases.extend(json.load(f))

    custom_paths = sorted(glob.glob("data/custom_cases/*.json"))
    for cp in custom_paths:
        with open(cp, "r") as f:
            cases.append(json.load(f))

    print(f"Loaded {len(cases)} cases for evaluation.")

    try:
        workflow = create_graph()
    except Exception as e:
        print(f"Failed to initialize workflow: {e}")
        return

    correct = 0
    total = 0
    results = []

    for case_data in cases:
        case_id = case_data["case_id"]

        # Determine expected outcome
        expected_entry = expected_outcomes.get(case_id)
        expected = expected_entry["expected_decision"] if expected_entry else None

        if not expected and "CUST" in case_id:
            task_desc = case_data.get("task", "")
            for status in ["ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE",
                           "NOT_ADMISSIBLE", "NEEDS_REVIEW", "ADMISSIBLE"]:
                if status in task_desc:
                    expected = status
                    break

        if not expected:
            print(f"  Skipping {case_id} — no expected outcome defined.")
            continue

        print(f"\nEvaluating {case_id}... Expected: {expected}")
        t0 = time.time()
        try:
            initial_state = AgentState(case=ClaimCase(**case_data))
            final_state = workflow.invoke(initial_state)
            decision = final_state["decision"]
            confidence = final_state["confidence"]
            elapsed = round(time.time() - t0, 2)

            is_correct = decision == expected
            if is_correct:
                correct += 1
                print(f"  [PASS] Got {decision} (confidence: {confidence:.2f}) in {elapsed}s")
            else:
                print(f"  [FAIL] Got {decision} (expected {expected}, confidence: {confidence:.2f}) in {elapsed}s")

            total += 1

            # Collect result
            results.append({
                "case_id": case_id,
                "expected": expected,
                "actual": decision,
                "correct": is_correct,
                "confidence": confidence,
                "elapsed_seconds": elapsed,
                "key_findings": final_state["key_findings"],
                "missing_evidence": final_state["missing_evidence"],
                "num_citations": len(final_state["citations"]),
                "num_evidence_chunks": len(final_state["retrieved_evidence"]),
                "validation_status": final_state["validation"].status,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "case_id": case_id,
                "expected": expected,
                "actual": "ERROR",
                "correct": False,
                "error": str(e),
            })

        # Cooldown delay to prevent Groq API rate limit throttling
        time.sleep(12)

    # Summary
    print("\n" + "=" * 60)
    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")

        # Retrieval quality
        avg_chunks = sum(r.get("num_evidence_chunks", 0) for r in results if "num_evidence_chunks" in r) / max(total, 1)
        avg_citations = sum(r.get("num_citations", 0) for r in results if "num_citations" in r) / max(total, 1)
        print(f"Avg evidence chunks retrieved: {avg_chunks:.1f}")
        print(f"Avg citations per decision: {avg_citations:.1f}")

        # Abstention cases
        abstention_cases = [r for r in results if r.get("actual") == "NEEDS_REVIEW"]
        print(f"Abstention (NEEDS_REVIEW) cases: {len(abstention_cases)}")

        # Failures
        failures = [r for r in results if not r.get("correct", True)]
        if failures:
            print(f"\nFailure Analysis ({len(failures)} cases):")
            for fail in failures:
                print(f"  {fail['case_id']}: expected {fail['expected']}, got {fail['actual']}")
    else:
        print("No evaluations performed.")

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    output_path = "evaluation/evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy_pct": round(correct / max(total, 1) * 100, 1),
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    evaluate()
