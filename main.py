# Your existing pipeline
import time
from classification_layer import AddressClassifier
from pre_processing import (
    expand_abbreviations,
    normalize_input,
    suggest_address_components,
)
from typing import Dict, List
import json
import csv
import os


def load_test_cases(filename: str) -> List[Dict]:
    """Load test cases from JSON file"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found, using empty list")
        return []


def evaluate_prediction(predicted: Dict, expected: Dict) -> Dict:
    """Evaluate a single prediction against expected results"""
    result = {
        "province_correct": False,
        "district_correct": False,
        "ward_correct": False,
        "province_match": None,
        "district_match": None,
        "ward_match": None,
    }

    # Get and title-case all predicted values first
    predicted_provinces = predicted.get("province", {}).get("classified", [])
    if predicted_provinces and predicted_provinces[0]:
        predicted_province = predicted_provinces[0].title().strip()
        result["province_match"] = predicted_province

    predicted_districts = predicted.get("district", {}).get("classified", [])
    if predicted_districts and predicted_districts[0]:
        predicted_district = predicted_districts[0].title().strip()
        result["district_match"] = predicted_district

    predicted_wards = predicted.get("ward", {}).get("classified", [])
    if predicted_wards and predicted_wards[0]:
        predicted_ward = predicted_wards[0].title().strip()
        result["ward_match"] = predicted_ward

    # Now, perform comparisons if expected values exist
    # Check province
    if expected.get("province"):
        expected_province = expected["province"].title().strip()
        if result["province_match"] and result["province_match"] == expected_province:
            result["province_correct"] = True

    # Check district
    if expected.get("district"):
        expected_district = expected["district"].title().strip()
        if result["district_match"] and result["district_match"] == expected_district:
            result["district_correct"] = True

    # Check ward
    if expected.get("ward"):
        expected_ward = expected["ward"].title().strip()
        if result["ward_match"] and result["ward_match"] == expected_ward:
            result["ward_correct"] = True

    return result


def safe_get_first(data_dict: Dict, component: str) -> str:
    """Safely get the first classified item or return None"""
    classified = data_dict.get(component, {}).get("classified", [])
    return classified[0] if classified and classified[0] else None


def run_comprehensive_test(
    test_file_path: str = "test.json", max_display: int = 10
) -> Dict:
    """
    Run comprehensive test suite matching main.py pattern with timing and export
    """
    # Load test cases
    test_cases = load_test_cases(test_file_path)
    total_cases = len(test_cases)

    classifier = AddressClassifier(
        "./data/provinces.txt", "./data/districts.txt", "./data/wards.txt"
    )

    if total_cases == 0:
        return {"error": "No test cases found"}

    print(f"\nTesting address classification with {total_cases} test cases:")
    print("=" * 80)

    # Initialize tracking variables
    correct_predictions = 0
    total_time = 0
    max_time = 0
    failed_cases = []
    all_test_results = []

    # Process test cases
    for i, test_case in enumerate(test_cases):
        address = test_case["text"]
        expected = test_case["result"]
        notes = test_case.get("notes", "")

        # Time the prediction
        start_time = time.perf_counter()
        normalized = normalize_input(address)
        expanded = expand_abbreviations(normalized)
        suggestions = suggest_address_components(expanded)
        print("Address Suggestions:", suggestions)
        raw_prediction = classifier.classify_address(
            suggestions
        )  # Rename to raw_prediction

        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000  # Convert to ms
        total_time += processing_time
        max_time = max(max_time, processing_time)

        predicted = {
            "province": {
                "classified": [raw_prediction.province]
                if raw_prediction.province
                else []
            },
            "district": {
                "classified": [raw_prediction.district]
                if raw_prediction.district
                else []
            },
            "ward": {
                "classified": [raw_prediction.ward] if raw_prediction.ward else []
            },
        }

        # Evaluate prediction
        evaluation = evaluate_prediction(predicted, expected)

        # Check if prediction matches expected result (all components correct)
        is_correct = (
            evaluation["province_correct"]
            and evaluation["district_correct"]
            and evaluation["ward_correct"]
        )

        if is_correct:
            correct_predictions += 1

        # Prepare result data matching main.py format
        reasons = []
        pred_province = evaluation["province_match"]
        pred_district = evaluation["district_match"]
        pred_ward = evaluation["ward_match"]

        if not evaluation["province_correct"]:
            reasons.append("province mismatch")
        if not evaluation["district_correct"]:
            reasons.append("district mismatch")
        if not evaluation["ward_correct"]:
            reasons.append("ward mismatch")
        if processing_time > 100:
            reasons.append(f"time_exceeded ({processing_time:.2f} ms > 100 ms)")

        fail_reason = "; ".join(reasons) if reasons else ""

        test_result = {
            "index": i + 1,
            "input": address,
            "normalized": normalized,
            "notes": notes,
            "expected_province": expected.get("province"),
            "expected_district": expected.get("district"),
            "expected_ward": expected.get("ward"),
            "got_province": pred_province,
            "got_district": pred_district,
            "got_ward": pred_ward,
            "time_ms": round(processing_time, 3),
            "status": "PASS" if is_correct else "FAIL",
            "fail_reason": fail_reason,
        }

        all_test_results.append(test_result)

        if not is_correct:
            failed_cases.append(test_result)

        # Show results for first few test cases or if incorrect
        if i < max_display or not is_correct:
            print(f"\nTest {i + 1}: {address}")
            print(f"Expected: {expected}")
            print(
                f"Predicted: {{'province': '{pred_province}', 'district': '{pred_district}', 'ward': '{pred_ward}'}}"
            )
            print(f"Correct: {'✅' if is_correct else '❌'}")
            print(f"Processing time: {processing_time:.2f}ms")
            if notes:
                print(f"Notes: {notes}")

        # Check performance requirements
        if processing_time > 100:  # 0.1s = 100ms
            print("⚠️  WARNING: Exceeds maximum time requirement!")
        elif processing_time > 10:  # 0.01s = 10ms
            print("⚠️  WARNING: Exceeds average time requirement!")

    # Summary statistics
    accuracy = (correct_predictions / total_cases) * 100
    avg_time = total_time / total_cases

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total test cases: {total_cases}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average processing time: {avg_time:.2f}ms")
    print(f"Maximum processing time: {max_time:.2f}ms")

    print("\n" + "=" * 80)
    print("PERFORMANCE CONSTRAINTS")
    print("=" * 80)
    print(f"Average time ≤ 10ms: {'✓' if avg_time <= 10 else '✗'} ({avg_time:.2f} ms)")
    print(f"Max time ≤ 100ms: {'✓' if max_time <= 100 else '✗'} ({max_time:.2f} ms)")

    if avg_time <= 10 and max_time <= 100:
        print("✅ All performance requirements met!")
    else:
        print("⚠️  Performance requirements not met!")

    return {
        "total_cases": total_cases,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "max_time": max_time,
        "failed_cases": failed_cases,
        "all_test_results": all_test_results,
    }


def export_test_results(test_results: Dict):
    """Export test results to CSV and JSON files matching main.py format"""
    if "error" in test_results:
        print("Cannot export results - test failed to run")
        return

    all_test_results = test_results["all_test_results"]
    failed_cases = test_results["failed_cases"]

    print("\n" + "=" * 80)
    print("EXPORTING TEST RESULTS")
    print("=" * 80)

    try:
        # CSV headers matching main.py format
        csv_headers = [
            "index",
            "input",
            "normalized",
            "notes",
            "expected_province",
            "expected_district",
            "expected_ward",
            "got_province",
            "got_district",
            "got_ward",
            "time_ms",
            "status",
            "fail_reason",
        ]

        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Export full test report
        test_report_path = os.path.join(output_dir, "_test_report.csv")
        with open(test_report_path, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_headers)
            writer.writeheader()
            for row in all_test_results:
                writer.writerow(row)
        print(
            f"Wrote test report CSV: {test_report_path} ({len(all_test_results)} rows)"
        )

        # Export failed cases only
        failed_cases_path = os.path.join(output_dir, "_failed_cases.csv")
        with open(failed_cases_path, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_headers)
            writer.writeheader()
            for row in failed_cases:
                writer.writerow(row)
        print(f"Wrote failed cases CSV: {failed_cases_path} ({len(failed_cases)} rows)")

        # Export JSON format for full report
        json_report_path = os.path.join(output_dir, "_test_report.json")
        with open(json_report_path, "w", encoding="utf-8") as jf:
            json.dump(all_test_results, jf, ensure_ascii=False, indent=2)
        print(
            f"Wrote test report JSON: {json_report_path} ({len(all_test_results)} rows)"
        )

    except Exception as e:
        print(f"Failed to export test results: {e}")


def show_failed_cases_summary(test_results: Dict, max_show: int = 10):
    """Display detailed information about failed test cases"""
    if "error" in test_results:
        return

    failed_cases = test_results.get("failed_cases", [])

    if not failed_cases:
        print("🎉 No failed cases!")
        return

    print(f"\nFAILED CASES SUMMARY (showing first {min(max_show, len(failed_cases))}):")
    print("=" * 80)

    for i, case in enumerate(failed_cases[:max_show]):
        print(f"\n[{case['index']}] {case['input']}")
        print(
            f"Expected - P: {case['expected_province'] or 'N/A'}, D: {case['expected_district'] or 'N/A'}, W: {case['expected_ward'] or 'N/A'}"
        )
        print(
            f"Got      - P: {case['got_province'] or 'N/A'}, D: {case['got_district'] or 'N/A'}, W: {case['got_ward'] or 'N/A'}"
        )
        print(f"Time: {case['time_ms']}ms | Issues: {case['fail_reason']}")

    if len(failed_cases) > max_show:
        print(f"\n... and {len(failed_cases) - max_show} more failed cases")


if __name__ == "__main__":
    test_results = run_comprehensive_test("./data/all_test.json", max_display=5)

    if "error" not in test_results:
        show_failed_cases_summary(test_results, max_show=10)
        export_test_results(test_results)
