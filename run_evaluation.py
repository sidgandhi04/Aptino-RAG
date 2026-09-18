"""
Evaluate all 12 public test cases against expected decisions.
Handles Groq rate limits with delays between cases.
Logs all results to evaluation/case_comparison.json and prints a comparison table.
"""
import json
import os
import sys
import time
import traceback

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.graph import create_graph
from agents.state import AgentState, ClaimCase


def load_cases():
    path = "Given Data/candidate_data/public_test_cases.json"
    with open(path, "r") as f:
        return json.load(f)


def load_expected():
    path = "data/expected_outcomes.json"
    with open(path, "r") as f:
        data = json.load(f)
        return {d["case_id"]: d["expected_decision"] for d in data}


def run_evaluation():
    print("=" * 70)
    print("  Aptino RAG -- Full Evaluation (12 Public Cases)")
    print("=" * 70)

    cases = load_cases()
    expected = load_expected()

    print(f"\nLoaded {len(cases)} cases, {len(expected)} expected outcomes.")
    print("Initializing workflow...")

    try:
        workflow = create_graph()
    except Exception as e:
        print(f"FATAL: Failed to initialize workflow: {e}")
        return

    print("Workflow ready.\n")

    results = []
    DELAY_BETWEEN_CASES = 15  # seconds - spread across TPM window

    for i, case_data in enumerate(cases):
        case_id = case_data["case_id"]
        exp = expected.get(case_id, "UNKNOWN")

        print(f"\n{'-' * 60}")
        print(f"  [{i+1}/{len(cases)}] {case_id}  |  Expected: {exp}")
        print(f"{'-' * 60}")

        t0 = time.time()
        try:
            initial_state = AgentState(case=ClaimCase(**case_data))
            final_state = workflow.invoke(initial_state)

            decision = final_state["decision"]
            confidence = final_state["confidence"]
            elapsed = round(time.time() - t0, 2)
            match = "PASS" if decision == exp else "FAIL"

            print(f"  Decision:   {decision}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Expected:   {exp}")
            print(f"  Result:     {match}")
            print(f"  Time:       {elapsed}s")

            result = {
                "case_id": case_id,
                "expected": exp,
                "actual": decision,
                "match": decision == exp,
                "confidence": confidence,
                "elapsed_seconds": elapsed,
                "key_findings": final_state["key_findings"],
                "applicable_limits": final_state["applicable_limits"],
                "missing_evidence": final_state["missing_evidence"],
                "missing_fields": final_state["missing_fields"],
                "num_citations": len(final_state["citations"]),
                "num_evidence_chunks": len(final_state["retrieved_evidence"]),
                "validation_status": final_state["validation"].status,
                "validation_attempts": final_state["validation_attempts"],
                "timings": final_state.get("timings", {}),
                "trace": final_state["trace"],
            }
            results.append(result)

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "case_id": case_id,
                "expected": exp,
                "actual": "ERROR",
                "match": False,
                "error": str(e),
                "elapsed_seconds": elapsed,
            })

        # Rate limit delay (skip after last case)
        if i < len(cases) - 1:
            print(f"\n  Waiting {DELAY_BETWEEN_CASES}s for rate limit cooldown...")
            time.sleep(DELAY_BETWEEN_CASES)

    # -- Summary Table --
    print("\n\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"\n  {'Case ID':<10} {'Expected':<28} {'Actual':<28} {'Match':<8} {'Conf':<6} {'Time':<6}")
    print(f"  {'-'*10} {'-'*28} {'-'*28} {'-'*8} {'-'*6} {'-'*6}")

    correct = 0
    total = len(results)

    for r in results:
        case_id = r["case_id"]
        exp = r["expected"]
        act = r["actual"]
        match_str = "OK" if r["match"] else "XX"
        conf = f"{r.get('confidence', 0):.2f}" if "confidence" in r else "ERR"
        elapsed = f"{r.get('elapsed_seconds', 0):.0f}s"
        if r["match"]:
            correct += 1
        print(f"  {case_id:<10} {exp:<28} {act:<28} {match_str:<8} {conf:<6} {elapsed:<6}")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  {'-'*88}")
    print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Failures detail
    failures = [r for r in results if not r["match"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    {f['case_id']}: expected={f['expected']}, got={f['actual']}")
            if "key_findings" in f:
                for kf in f.get("key_findings", [])[:2]:
                    print(f"      -> {kf[:120]}")

    # -- Save to file --
    os.makedirs("evaluation", exist_ok=True)
    output_path = "evaluation/case_comparison.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump({
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy_pct": round(accuracy, 1),
                "total_time_seconds": round(sum(r.get("elapsed_seconds", 0) for r in results), 1),
            },
            "results": results,
        }, fp, indent=2, default=str, ensure_ascii=False)

    print(f"\n  Full results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
