"""
Datasheet ETL: Comprehensive Test Suite

This module provides comprehensive tests for the Vision-first + Double Verification
pipeline, demonstrating various scenarios and edge cases.

Author: Axiom
Date: 2026-02-13
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from double_verifier import (
    DoubleVerifier, 
    VerificationStatus, 
    ConflictSeverity,
    NumericComparator,
    UnitComparator
)
from pipeline import DatasheetPipeline, PipelineConfig


def test_numeric_comparator():
    """Test numeric comparison with tolerance."""
    print("\n" + "=" * 60)
    print("TEST: Numeric Comparator")
    print("=" * 60)
    
    comp = NumericComparator(tolerance=0.01)
    
    test_cases = [
        ("3.3", "3.3", True, "Exact match"),
        ("3.3", "3.30", True, "Trailing zero"),
        ("3.3", "3.33", True, "Within 1% tolerance"),
        ("3.3", "3.5", False, "Outside tolerance"),
        ("0", "0", True, "Zero values"),
        ("100", "101", True, "1% of 100"),
        ("100", "102", False, "2% of 100"),
        ("1.5V", "1.5", True, "With unit suffix"),
        ("-5", "-5.0", True, "Negative values"),
    ]
    
    passed = 0
    for v1, v2, expected, desc in test_cases:
        result = comp.compare(v1, v2)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {desc}: '{v1}' vs '{v2}' = {result}")
        if result == expected:
            passed += 1
    
    print(f"\nResult: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_unit_comparator():
    """Test unit normalization and comparison."""
    print("\n" + "=" * 60)
    print("TEST: Unit Comparator")
    print("=" * 60)
    
    comp = UnitComparator()
    
    test_cases = [
        ("µA", "uA", True, "Micro symbol variants"),
        ("µA", "μA", True, "Different micro characters"),
        ("mA", "milliamp", True, "Abbreviation vs full"),
        ("V", "volt", True, "V vs volt"),
        ("kHz", "kilohertz", True, "kHz vs kilohertz"),
        ("°C", "C", True, "Degree symbol"),
        ("MHz", "mhz", True, "Case insensitive"),
        ("V", "A", False, "Different units"),
        ("mV", "V", False, "Different scale"),
    ]
    
    passed = 0
    for v1, v2, expected, desc in test_cases:
        result = comp.compare(v1, v2)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {desc}: '{v1}' vs '{v2}' = {result}")
        if result == expected:
            passed += 1
    
    print(f"\nResult: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_double_verification_perfect_match():
    """Test verification with perfect match."""
    print("\n" + "=" * 60)
    print("TEST: Double Verification - Perfect Match")
    print("=" * 60)
    
    verifier = DoubleVerifier()
    
    vision_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Min", "Max", "Unit"],
        "rows": [
            {"Parameter": "VIN", "Min": "4.2", "Max": "36", "Unit": "V"},
            {"Parameter": "IOUT", "Min": "", "Max": "3", "Unit": "A"},
        ]
    }
    
    rule_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Min", "Max", "Unit"],
        "rows": [
            {"Parameter": "VIN", "Min": "4.2", "Max": "36", "Unit": "V"},
            {"Parameter": "IOUT", "Min": "", "Max": "3", "Unit": "A"},
        ]
    }
    
    result = verifier.verify(vision_result, rule_result)
    
    print(f"  Status: {result.status.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Conflicts: {len(result.conflicts)}")
    
    success = (
        result.status == VerificationStatus.VERIFIED and
        result.confidence == 1.0 and
        len(result.conflicts) == 0
    )
    
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_double_verification_unit_conflict():
    """Test verification with unit format conflict."""
    print("\n" + "=" * 60)
    print("TEST: Double Verification - Unit Conflict")
    print("=" * 60)
    
    verifier = DoubleVerifier()
    
    vision_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Value", "Unit"],
        "rows": [
            {"Parameter": "IQ", "Value": "25", "Unit": "µA"},
        ]
    }
    
    rule_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Value", "Unit"],
        "rows": [
            {"Parameter": "IQ", "Value": "25", "Unit": "uA"},  # Different format
        ]
    }
    
    result = verifier.verify(vision_result, rule_result)
    
    print(f"  Status: {result.status.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Conflicts: {len(result.conflicts)}")
    
    # Should recognize µA and uA as the same
    success = (
        result.status == VerificationStatus.VERIFIED and
        result.confidence >= 0.95
    )
    
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_double_verification_numeric_conflict():
    """Test verification with numeric value conflict."""
    print("\n" + "=" * 60)
    print("TEST: Double Verification - Numeric Conflict")
    print("=" * 60)
    
    verifier = DoubleVerifier()
    
    vision_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Value"],
        "rows": [
            {"Parameter": "VIN_MAX", "Value": "36"},
        ]
    }
    
    rule_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Value"],
        "rows": [
            {"Parameter": "VIN_MAX", "Value": "35"},  # Different value!
        ]
    }
    
    result = verifier.verify(vision_result, rule_result)
    
    print(f"  Status: {result.status.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Conflicts: {len(result.conflicts)}")
    
    for c in result.conflicts:
        print(f"    - {c.field}: '{c.vision_value}' vs '{c.rule_value}' "
              f"[{c.severity.value}] -> {c.resolution}")
    
    # Should detect conflict and resolve (Vision wins for numeric)
    success = (
        len(result.conflicts) > 0 and
        result.conflicts[0].resolution == "vision_wins"
    )
    
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_double_verification_critical_param():
    """Test verification with critical parameter conflict."""
    print("\n" + "=" * 60)
    print("TEST: Double Verification - Critical Parameter")
    print("=" * 60)
    
    verifier = DoubleVerifier(confidence_threshold=0.60)
    
    vision_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Value"],
        "rows": [
            {"Parameter": "Vin_max", "Value": "36"},
            {"Parameter": "Iout_max", "Value": "3"},
        ]
    }
    
    rule_result = {
        "table_id": "test_1",
        "headers": ["Parameter", "Value"],
        "rows": [
            {"Parameter": "Vin_max", "Value": "24"},  # Big difference in critical param!
            {"Parameter": "Iout_max", "Value": "3"},
        ]
    }
    
    result = verifier.verify(vision_result, rule_result)
    
    print(f"  Status: {result.status.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Conflicts: {len(result.conflicts)}")
    
    for c in result.conflicts:
        print(f"    - {c.field}: '{c.vision_value}' vs '{c.rule_value}' "
              f"[{c.severity.value}]")
    
    # Critical parameter conflict should trigger needs_review
    success = result.status == VerificationStatus.NEEDS_REVIEW
    
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_pipeline_end_to_end():
    """Test complete pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("TEST: Pipeline End-to-End")
    print("=" * 60)
    
    config = PipelineConfig(
        vision_model="gpt-4o-mini",
        verification_enabled=True,
        include_conflicts=True
    )
    
    pipeline = DatasheetPipeline(config)
    
    # Process mock datasheet
    result = pipeline.process("test.pdf")
    
    print(f"  Status: {result['verification']['status']}")
    print(f"  Confidence: {result['verification']['confidence']:.0%}")
    print(f"  Tables extracted: {len(result['tables'])}")
    print(f"  Parameters extracted: {len(result['parameters'])}")
    
    # Check stats
    stats = pipeline.get_stats()
    print(f"  Vision calls: {stats['vision_stats']['calls']}")
    print(f"  Estimated cost: ${stats['vision_stats']['estimated_cost']:.4f}")
    
    success = (
        result['verification']['status'] == 'verified' and
        result['verification']['confidence'] >= 0.90 and
        len(result['parameters']) > 0
    )
    
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("DATASHEET ETL - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Numeric Comparator", test_numeric_comparator),
        ("Unit Comparator", test_unit_comparator),
        ("Perfect Match", test_double_verification_perfect_match),
        ("Unit Conflict", test_double_verification_unit_conflict),
        ("Numeric Conflict", test_double_verification_numeric_conflict),
        ("Critical Parameter", test_double_verification_critical_param),
        ("Pipeline E2E", test_pipeline_end_to_end),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
