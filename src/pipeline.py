"""
Datasheet ETL: Complete Pipeline v0.4

This module provides the complete Vision-first + Double Verification pipeline
for extracting structured data from semiconductor datasheets.

Features:
- Vision-first extraction (primary)
- Rule-based extraction (secondary)
- Double verification with conflict resolution
- Confidence scoring and review queue
- Modular and extensible design

Usage:
    python pipeline.py datasheets/lmr51430.pdf output/

Author: Axiom
Date: 2026-02-13
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Local imports
from double_verifier import DoubleVerifier, VerificationStatus, VerificationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Extraction
    primary_engine: str = "vision"
    secondary_engine: str = "rule"
    vision_model: str = "gpt-4o-mini"
    
    # Verification
    verification_enabled: bool = True
    numeric_tolerance: float = 0.01
    confidence_threshold: float = 0.60
    
    # Output
    output_format: str = "json"
    include_raw: bool = False
    include_conflicts: bool = True
    
    # Cost control
    budget_per_datasheet: float = 0.50
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Mock Extractors (for demo without API keys)
# ============================================================================

class MockVisionExtractor:
    """
    Mock Vision extractor for demonstration.
    
    In production, this would call GPT-4o-mini or similar Vision API.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._call_count = 0
    
    def extract_table(self, image_path: str, context: Dict = None) -> Dict[str, Any]:
        """Extract table from image using Vision model."""
        self._call_count += 1
        
        # Mock response simulating Vision extraction
        # In production: call OpenAI/Anthropic/Google Vision API
        return {
            "table_id": f"vision_table_{self._call_count}",
            "title": "Electrical Characteristics",
            "headers": ["Parameter", "Test Condition", "Min", "Typ", "Max", "Unit"],
            "rows": [
                {
                    "Parameter": "Input Voltage Range",
                    "Test Condition": "",
                    "Min": "4.2",
                    "Typ": "",
                    "Max": "36",
                    "Unit": "V"
                },
                {
                    "Parameter": "Quiescent Current",
                    "Test Condition": "VFB = 1.1V",
                    "Min": "",
                    "Typ": "25",
                    "Max": "40",
                    "Unit": "µA"
                },
                {
                    "Parameter": "Output Current",
                    "Test Condition": "Continuous",
                    "Min": "",
                    "Typ": "",
                    "Max": "3",
                    "Unit": "A"
                },
                {
                    "Parameter": "Switching Frequency",
                    "Test Condition": "",
                    "Min": "360",
                    "Typ": "400",
                    "Max": "440",
                    "Unit": "kHz"
                },
                {
                    "Parameter": "Efficiency",
                    "Test Condition": "VIN=12V, VOUT=5V, IOUT=2A",
                    "Min": "",
                    "Typ": "92",
                    "Max": "",
                    "Unit": "%"
                }
            ],
            "confidence": 0.95,
            "extraction_method": "vision",
            "model": self.model,
            "cost_usd": 0.002
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self._call_count,
            "estimated_cost": self._call_count * 0.002
        }


class MockRuleExtractor:
    """
    Mock Rule-based extractor for demonstration.
    
    In production, this would use pdfplumber + heuristics.
    """
    
    def __init__(self):
        self._call_count = 0
    
    def extract_table(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """Extract table using rule-based methods."""
        self._call_count += 1
        
        # Mock response simulating rule-based extraction
        # Intentionally slightly different to test verification
        return {
            "table_id": f"rule_table_{self._call_count}",
            "title": "Electrical Characteristics",
            "headers": ["Parameter", "Test Condition", "Min", "Typ", "Max", "Unit"],
            "rows": [
                {
                    "Parameter": "Input Voltage Range",
                    "Test Condition": "",
                    "Min": "4.2",
                    "Typ": "",
                    "Max": "36",
                    "Unit": "V"
                },
                {
                    "Parameter": "Quiescent Current",
                    "Test Condition": "VFB = 1.1V",
                    "Min": "",
                    "Typ": "25",
                    "Max": "40",
                    "Unit": "uA"  # Different format: uA vs µA
                },
                {
                    "Parameter": "Output Current",
                    "Test Condition": "Continuous",
                    "Min": "",
                    "Typ": "",
                    "Max": "3.0",  # Slightly different: 3.0 vs 3
                    "Unit": "A"
                },
                {
                    "Parameter": "Switching Frequency",
                    "Test Condition": "",
                    "Min": "360",
                    "Typ": "400",
                    "Max": "440",
                    "Unit": "kHz"
                },
                {
                    "Parameter": "Efficiency",
                    "Test Condition": "VIN=12V, VOUT=5V, IOUT=2A",
                    "Min": "",
                    "Typ": "92",
                    "Max": "",
                    "Unit": "%"
                }
            ],
            "confidence": 0.85,
            "extraction_method": "rule_based"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "calls": self._call_count
        }


# ============================================================================
# Pipeline
# ============================================================================

class DatasheetPipeline:
    """
    Complete Datasheet ETL Pipeline.
    
    Implements Vision-first + Double Verification architecture.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize extractors
        self.vision_extractor = MockVisionExtractor(model=self.config.vision_model)
        self.rule_extractor = MockRuleExtractor()
        
        # Initialize verifier
        self.verifier = DoubleVerifier(
            numeric_tolerance=self.config.numeric_tolerance,
            confidence_threshold=self.config.confidence_threshold
        )
        
        # Stats
        self._processed_count = 0
        self._review_queue: List[Dict] = []
    
    def process(
        self,
        pdf_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single datasheet.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path for output JSON
            
        Returns:
            Complete extraction result with verification status
        """
        logger.info(f"Processing: {pdf_path}")
        self._processed_count += 1
        
        # Step 1: Vision extraction (primary)
        logger.info("Step 1: Vision extraction...")
        vision_result = self.vision_extractor.extract_table(pdf_path)
        
        # Step 2: Rule-based extraction (secondary)
        logger.info("Step 2: Rule-based extraction...")
        rule_result = self.rule_extractor.extract_table(pdf_path)
        
        # Step 3: Double verification
        logger.info("Step 3: Double verification...")
        verification = self.verifier.verify(vision_result, rule_result)
        
        # Step 4: Build final result
        result = self._build_result(
            pdf_path, vision_result, rule_result, verification
        )
        
        # Step 5: Handle review queue
        if verification.status == VerificationStatus.NEEDS_REVIEW:
            self._review_queue.append({
                "pdf_path": pdf_path,
                "result": result,
                "conflicts": [asdict(c) for c in verification.conflicts]
            })
            logger.warning(f"Added to review queue: {pdf_path}")
        
        # Step 6: Save output
        if output_path:
            self._save_output(result, output_path)
            logger.info(f"Saved to: {output_path}")
        
        return result
    
    def _build_result(
        self,
        pdf_path: str,
        vision_result: Dict,
        rule_result: Dict,
        verification: VerificationResult
    ) -> Dict[str, Any]:
        """Build the final result structure."""
        result = {
            "metadata": {
                "source_file": os.path.basename(pdf_path),
                "pipeline_version": "0.4",
                "extraction_method": "vision_first_double_verified",
                "vision_model": self.config.vision_model
            },
            "verification": {
                "status": verification.status.value,
                "confidence": round(verification.confidence, 3),
                "resolution_method": verification.resolution_method,
                "conflict_count": len(verification.conflicts)
            },
            "tables": [verification.final_result],
            "parameters": self._extract_parameters(verification.final_result)
        }
        
        # Include conflicts if configured
        if self.config.include_conflicts and verification.conflicts:
            result["conflicts"] = [
                {
                    "field": c.field,
                    "vision_value": c.vision_value,
                    "rule_value": c.rule_value,
                    "severity": c.severity.value,
                    "resolution": c.resolution,
                    "resolved_value": c.resolved_value
                }
                for c in verification.conflicts
            ]
        
        # Include raw results if configured
        if self.config.include_raw:
            result["raw"] = {
                "vision": vision_result,
                "rule": rule_result
            }
        
        return result
    
    def _extract_parameters(self, table: Dict) -> Dict[str, Any]:
        """Extract key parameters from table."""
        params = {}
        
        rows = table.get("rows", [])
        for row in rows:
            param_name = row.get("Parameter", "").lower().replace(" ", "_")
            
            # Extract numeric values
            for col in ["Min", "Typ", "Max"]:
                val = row.get(col, "")
                if val:
                    try:
                        params[f"{param_name}_{col.lower()}"] = float(val)
                    except ValueError:
                        pass
            
            # Extract unit
            unit = row.get("Unit", "")
            if unit:
                params[f"{param_name}_unit"] = unit
        
        return params
    
    def _save_output(self, result: Dict, output_path: str):
        """Save result to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def get_review_queue(self) -> List[Dict]:
        """Get items in the review queue."""
        return self._review_queue.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "processed_count": self._processed_count,
            "review_queue_size": len(self._review_queue),
            "vision_stats": self.vision_extractor.get_stats(),
            "rule_stats": self.rule_extractor.get_stats(),
            "verifier_stats": self.verifier.get_stats()
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_path> [output_dir]")
        print("\nExample:")
        print("  python pipeline.py datasheets/lmr51430.pdf output/")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    # Create output path
    pdf_name = Path(pdf_path).stem
    output_path = os.path.join(output_dir, f"{pdf_name}_v04.json")
    
    # Run pipeline
    pipeline = DatasheetPipeline()
    result = pipeline.process(pdf_path, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Source: {pdf_path}")
    print(f"Output: {output_path}")
    print(f"Status: {result['verification']['status']}")
    print(f"Confidence: {result['verification']['confidence']:.1%}")
    print(f"Conflicts: {result['verification']['conflict_count']}")
    
    if result.get('conflicts'):
        print("\nConflicts resolved:")
        for c in result['conflicts']:
            print(f"  - {c['field']}: '{c['vision_value']}' vs '{c['rule_value']}' "
                  f"-> {c['resolution']}")
    
    print("\nParameters extracted:")
    for key, value in result.get('parameters', {}).items():
        print(f"  - {key}: {value}")
    
    # Print stats
    stats = pipeline.get_stats()
    print(f"\nStats:")
    print(f"  - Vision calls: {stats['vision_stats']['calls']}")
    print(f"  - Estimated cost: ${stats['vision_stats']['estimated_cost']:.4f}")
    print(f"  - Review queue: {stats['review_queue_size']} items")


if __name__ == "__main__":
    main()
