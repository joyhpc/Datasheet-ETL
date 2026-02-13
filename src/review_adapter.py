"""
Datasheet ETL: Review UI Adapter

Converts ETL output format to Review UI expected format.
This enables seamless integration between Axiom's ETL and Sirius's Review UI.

Author: Axiom
Date: 2026-02-14
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


# Parameter name mapping: ETL -> Review UI
PARAM_MAPPING = {
    "input_voltage_range_min": "v_in_min",
    "input_voltage_range_max": "v_in_max",
    "output_voltage_min": "v_out_min",
    "output_voltage_max": "v_out_max",
    "output_current_max": "i_out_max",
    "quiescent_current_typ": "i_q",
    "quiescent_current_max": "i_q_max",
    "switching_frequency_typ": "fsw",
    "switching_frequency_min": "fsw_min",
    "switching_frequency_max": "fsw_max",
    "efficiency_typ": "efficiency",
}

# Confidence thresholds for needs_review
REVIEW_THRESHOLD = 0.95


def convert_etl_to_review_format(etl_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ETL output format to Review UI expected format.
    
    Args:
        etl_output: ETL extraction result (from lmr51430_v04.json format)
        
    Returns:
        Review UI compatible format
    """
    metadata = etl_output.get("metadata", {})
    verification = etl_output.get("verification", {})
    parameters = etl_output.get("parameters", {})
    tables = etl_output.get("tables", [])
    
    # Extract MPN from source file
    source_file = metadata.get("source_file", "")
    mpn = Path(source_file).stem.upper() if source_file else "UNKNOWN"
    
    # Build params dict
    params = {}
    overall_confidence = verification.get("confidence", 0.9)
    extraction_method = metadata.get("extraction_method", "unknown")
    
    # Group parameters by base name
    param_groups = _group_parameters(parameters)
    
    for base_name, values in param_groups.items():
        # Map to Review UI name
        review_name = PARAM_MAPPING.get(base_name, base_name)
        
        # Get value and unit
        value = values.get("value")
        unit = values.get("unit", "")
        
        if value is None:
            continue
        
        # Determine confidence (use table confidence if available)
        param_confidence = _get_param_confidence(base_name, tables, overall_confidence)
        
        # Determine verification methods
        verified_by = _get_verification_methods(extraction_method)
        
        # Determine if needs review
        needs_review = param_confidence < REVIEW_THRESHOLD
        
        params[review_name] = {
            "value": value,
            "unit": unit,
            "confidence": param_confidence,
            "source": _get_source_reference(base_name, tables),
            "verified_by": verified_by,
            "needs_review": needs_review
        }
    
    return {
        "mpn": mpn,
        "manufacturer": _infer_manufacturer(mpn),
        "source_file": source_file,
        "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "params": params
    }


def _group_parameters(parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group flat parameters into structured format."""
    groups = {}
    
    for key, value in parameters.items():
        # Handle unit keys
        if key.endswith("_unit"):
            base_name = key[:-5]  # Remove "_unit"
            if base_name not in groups:
                groups[base_name] = {}
            groups[base_name]["unit"] = value
        # Handle value keys (min, typ, max)
        elif key.endswith("_min"):
            base_name = key[:-4] + "_min"
            if base_name not in groups:
                groups[base_name] = {}
            groups[base_name]["value"] = value
            # Copy unit from base if available
            base_unit_key = key[:-4] + "_unit"
            if base_unit_key in parameters:
                groups[base_name]["unit"] = parameters[base_unit_key]
        elif key.endswith("_typ"):
            base_name = key[:-4] + "_typ"
            if base_name not in groups:
                groups[base_name] = {}
            groups[base_name]["value"] = value
            base_unit_key = key[:-4] + "_unit"
            if base_unit_key in parameters:
                groups[base_name]["unit"] = parameters[base_unit_key]
        elif key.endswith("_max"):
            base_name = key[:-4] + "_max"
            if base_name not in groups:
                groups[base_name] = {}
            groups[base_name]["value"] = value
            base_unit_key = key[:-4] + "_unit"
            if base_unit_key in parameters:
                groups[base_name]["unit"] = parameters[base_unit_key]
    
    return groups


def _get_param_confidence(param_name: str, tables: List[Dict], default: float) -> float:
    """Get confidence for a specific parameter from tables."""
    # For now, use table confidence if available
    if tables:
        return tables[0].get("confidence", default)
    return default


def _get_verification_methods(extraction_method: str) -> List[str]:
    """Determine verification methods from extraction method."""
    methods = []
    
    if "vision" in extraction_method.lower():
        methods.append("vision")
    if "double" in extraction_method.lower() or "verified" in extraction_method.lower():
        methods.append("pdfplumber")
    if "rule" in extraction_method.lower():
        methods.append("rule_based")
    
    if not methods:
        methods.append("unknown")
    
    return methods


def _get_source_reference(param_name: str, tables: List[Dict]) -> str:
    """Generate source reference for a parameter."""
    if tables:
        table = tables[0]
        table_id = table.get("table_id", "table_1")
        return f"page_1_{table_id}"
    return "unknown"


def _infer_manufacturer(mpn: str) -> str:
    """Infer manufacturer from MPN prefix."""
    mpn_upper = mpn.upper()
    
    # Common manufacturer prefixes
    prefixes = {
        "LM": "Texas Instruments",
        "TPS": "Texas Instruments",
        "LMR": "Texas Instruments",
        "MAX": "Analog Devices",
        "AD": "Analog Devices",
        "LT": "Analog Devices",
        "LTC": "Analog Devices",
        "MCP": "Microchip",
        "PIC": "Microchip",
        "STM": "STMicroelectronics",
        "NCP": "onsemi",
        "MC": "onsemi",
        "ISL": "Renesas",
        "IR": "Infineon",
        "INA": "Texas Instruments",
        "OPA": "Texas Instruments",
    }
    
    for prefix, manufacturer in prefixes.items():
        if mpn_upper.startswith(prefix):
            return manufacturer
    
    return ""


def convert_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert an ETL output file to Review UI format.
    
    Args:
        input_path: Path to ETL output JSON file
        output_path: Path for Review UI format file (optional)
        
    Returns:
        Path to output file
    """
    with open(input_path, 'r') as f:
        etl_output = json.load(f)
    
    review_format = convert_etl_to_review_format(etl_output)
    
    if output_path is None:
        # Generate output path
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_review.json")
    
    with open(output_path, 'w') as f:
        json.dump(review_format, f, indent=2)
    
    return output_path


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python review_adapter.py <input_json> [output_json]")
        print("\nConverts ETL output to Review UI format.")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result_path = convert_file(input_path, output_path)
    print(f"Converted: {input_path} -> {result_path}")


if __name__ == "__main__":
    main()
