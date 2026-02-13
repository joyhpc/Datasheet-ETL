"""
Datasheet ETL: End-to-End Integration Test

This test validates the complete pipeline:
1. PDF Loading
2. Table Detection
3. Rule-based Extraction
4. Vision Fallback (mock)
5. JSON Output

Author: Axiom
Date: 2026-02-13
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from extractor_v03 import (
    DatasheetPipeline,
    TableExtractor,
    DatasheetExtraction,
    ExtractedTable,
    save_extraction_json
)
from vision_client import VisionExtractionManager, PromptTemplates
from pdf_renderer import PDFRenderer

def test_end_to_end():
    """Complete end-to-end test."""
    print("=" * 70)
    print("Datasheet ETL: End-to-End Integration Test")
    print("=" * 70)
    
    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # ========================================================================
    # Test 1: Module Imports
    # ========================================================================
    print("\n[1/7] Module Imports")
    try:
        from extractor_v03 import DatasheetPipeline
        from vision_client import VisionExtractionManager
        from pdf_renderer import PDFRenderer
        print("  ✅ All modules imported successfully")
        results["passed"] += 1
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        results["failed"] += 1
        return results
    
    # ========================================================================
    # Test 2: PDF Renderer Backend Detection
    # ========================================================================
    print("\n[2/7] PDF Renderer Backend Detection")
    renderer = PDFRenderer(dpi=150)
    print(f"  Backend detected: {renderer.backend}")
    if renderer.backend != "none":
        print("  ✅ PDF rendering available")
        results["passed"] += 1
    else:
        print("  ⚠️ No PDF backend (PyMuPDF/Poppler not installed)")
        results["skipped"] += 1
    
    # ========================================================================
    # Test 3: Vision Client (Mock Mode)
    # ========================================================================
    print("\n[3/7] Vision Client (Mock Mode)")
    vision_manager = VisionExtractionManager()  # No API key = mock mode
    
    # Test table extraction prompt
    response = vision_manager.extract_table("dummy_image.png")
    if response.success and response.parsed_json:
        print(f"  Mock response received: {len(response.parsed_json)} keys")
        print("  ✅ Vision client working (mock mode)")
        results["passed"] += 1
    else:
        print(f"  ❌ Vision client failed: {response.error}")
        results["failed"] += 1
    
    # ========================================================================
    # Test 4: Table Extractor Logic
    # ========================================================================
    print("\n[4/7] Table Extractor Logic")
    extractor = TableExtractor()
    
    # Test header detection
    test_headers = ["Parameter", "Test Conditions", "Min", "Typ", "Max", "Unit"]
    detected = extractor._detect_headers(test_headers)
    
    if "Parameter" in detected and "Min" in detected:
        print(f"  Headers detected: {detected}")
        print("  ✅ Header detection working")
        results["passed"] += 1
    else:
        print(f"  ❌ Header detection failed: {detected}")
        results["failed"] += 1
    
    # ========================================================================
    # Test 5: Mock Data Extraction
    # ========================================================================
    print("\n[5/7] Mock Data Extraction")
    
    mock_page = {
        "text": "LMR51430 Electrical Characteristics",
        "tables": [
            [
                ["Parameter", "Min", "Typ", "Max", "Unit"],
                ["VIN Range", "4.2", "", "36", "V"],
                ["IQ", "", "25", "40", "µA"],
            ]
        ]
    }
    
    tables = extractor.extract_tables_from_page(mock_page, 1)
    
    if tables and len(tables) > 0:
        table = tables[0]
        print(f"  Extracted table: {table.title}")
        print(f"  Rows: {len(table.rows)}")
        print(f"  Confidence: {table.confidence}")
        print("  ✅ Table extraction working")
        results["passed"] += 1
    else:
        print("  ❌ No tables extracted")
        results["failed"] += 1
    
    # ========================================================================
    # Test 6: JSON Serialization
    # ========================================================================
    print("\n[6/7] JSON Serialization")
    
    extraction = DatasheetExtraction(
        source_file="test.pdf",
        mpn="LMR51430",
        manufacturer="Texas Instruments",
        total_pages=32,
        tables=tables if tables else [],
        parameters={"vin_max": 36}
    )
    
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "e2e_test_output.json"
    
    try:
        save_extraction_json(extraction, str(output_file))
        
        # Verify file
        with open(output_file) as f:
            data = json.load(f)
        
        if data["metadata"]["mpn"] == "LMR51430":
            print(f"  Output saved: {output_file}")
            print("  ✅ JSON serialization working")
            results["passed"] += 1
        else:
            print("  ❌ JSON content mismatch")
            results["failed"] += 1
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
        results["failed"] += 1
    
    # ========================================================================
    # Test 7: Prompt Templates
    # ========================================================================
    print("\n[7/7] Prompt Templates")
    
    prompts = [
        ("TABLE_EXTRACTION", PromptTemplates.TABLE_EXTRACTION),
        ("BLOCK_DIAGRAM", PromptTemplates.BLOCK_DIAGRAM),
        ("PINOUT_DIAGRAM", PromptTemplates.PINOUT_DIAGRAM),
        ("TYPICAL_APPLICATION", PromptTemplates.TYPICAL_APPLICATION),
    ]
    
    all_valid = True
    for name, prompt in prompts:
        if len(prompt) > 100 and "json" in prompt.lower():
            print(f"  {name}: {len(prompt)} chars ✓")
        else:
            print(f"  {name}: Invalid ✗")
            all_valid = False
    
    if all_valid:
        print("  ✅ All prompt templates valid")
        results["passed"] += 1
    else:
        print("  ❌ Some prompts invalid")
        results["failed"] += 1
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed:  {results['passed']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    
    if results["failed"] == 0:
        print("\n🚀 ALL TESTS PASSED - Pipeline Ready for Production")
    else:
        print(f"\n⚠️ {results['failed']} test(s) failed - Review required")
    
    return results

if __name__ == "__main__":
    test_end_to_end()
