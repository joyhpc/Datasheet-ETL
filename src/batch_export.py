"""
Datasheet ETL: Batch Processor & Export Module

This module provides batch processing capabilities and export formats
for integration with Import Script and other downstream systems.

Features:
- Batch processing of multiple datasheets
- CSV export for Import Script compatibility
- Progress tracking and reporting
- Error handling and retry logic

Author: Axiom
Date: 2026-02-14
"""

import json
import csv
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BatchResult:
    """Result of batch processing."""
    total: int
    success: int
    failed: int
    skipped: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]]
    duration_seconds: float


@dataclass
class ExportConfig:
    """Configuration for export."""
    format: str = "json"  # "json", "csv", "both"
    include_metadata: bool = True
    include_raw: bool = False
    flatten_params: bool = True  # For CSV compatibility


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """
    Batch processing for multiple datasheets.
    
    Usage:
        processor = BatchProcessor(pipeline)
        result = processor.process_directory("datasheets/", "output/")
    """
    
    def __init__(
        self,
        pipeline,
        max_retries: int = 2,
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            pipeline: DatasheetPipeline instance
            max_retries: Maximum retries for failed extractions
            on_progress: Callback for progress updates (current, total, filename)
        """
        self.pipeline = pipeline
        self.max_retries = max_retries
        self.on_progress = on_progress
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.pdf"
    ) -> BatchResult:
        """
        Process all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output files
            pattern: Glob pattern for PDF files
            
        Returns:
            BatchResult with statistics and results
        """
        import glob
        import time
        
        start_time = time.time()
        
        # Find all PDF files
        pdf_files = list(Path(input_dir).glob(pattern))
        total = len(pdf_files)
        
        logger.info(f"Found {total} PDF files in {input_dir}")
        
        results = []
        errors = []
        success = 0
        failed = 0
        skipped = 0
        
        for i, pdf_path in enumerate(pdf_files):
            filename = pdf_path.name
            
            # Progress callback
            if self.on_progress:
                self.on_progress(i + 1, total, filename)
            
            logger.info(f"[{i+1}/{total}] Processing: {filename}")
            
            # Check if already processed
            output_path = Path(output_dir) / f"{pdf_path.stem}_v04.json"
            if output_path.exists():
                logger.info(f"  Skipping (already exists): {output_path}")
                skipped += 1
                continue
            
            # Process with retry
            result = self._process_with_retry(str(pdf_path), str(output_path))
            
            if result:
                results.append({
                    "file": filename,
                    "status": "success",
                    "output": str(output_path),
                    "confidence": result.get("verification", {}).get("confidence", 0)
                })
                success += 1
            else:
                errors.append({
                    "file": filename,
                    "error": "Failed after retries"
                })
                failed += 1
        
        duration = time.time() - start_time
        
        return BatchResult(
            total=total,
            success=success,
            failed=failed,
            skipped=skipped,
            results=results,
            errors=errors,
            duration_seconds=duration
        )
    
    def _process_with_retry(
        self,
        pdf_path: str,
        output_path: str
    ) -> Optional[Dict]:
        """Process a single file with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                result = self.pipeline.process(pdf_path, output_path)
                return result
            except Exception as e:
                logger.warning(f"  Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    logger.error(f"  All retries exhausted for {pdf_path}")
                    return None
        return None


# ============================================================================
# Export Functions
# ============================================================================

class Exporter:
    """
    Export extraction results to various formats.
    
    Supports:
    - JSON (default)
    - CSV (for Import Script compatibility)
    - Flattened parameters for spreadsheet use
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
    
    def export_to_csv(
        self,
        results: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Export results to CSV format.
        
        Args:
            results: List of extraction results
            output_path: Path for CSV file
            
        Returns:
            Path to created CSV file
        """
        if not results:
            logger.warning("No results to export")
            return ""
        
        # Collect all parameters across all results
        all_params = set()
        for result in results:
            params = result.get("parameters", {})
            all_params.update(params.keys())
        
        # Define CSV columns
        base_columns = ["source_file", "confidence", "status"]
        param_columns = sorted(list(all_params))
        columns = base_columns + param_columns
        
        # Write CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in results:
                row = {
                    "source_file": result.get("metadata", {}).get("source_file", ""),
                    "confidence": result.get("verification", {}).get("confidence", 0),
                    "status": result.get("verification", {}).get("status", ""),
                }
                
                # Add parameters
                params = result.get("parameters", {})
                for param in param_columns:
                    row[param] = params.get(param, "")
                
                writer.writerow(row)
        
        logger.info(f"Exported {len(results)} results to {output_path}")
        return output_path
    
    def export_for_import_script(
        self,
        results: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Export results in format compatible with Import Script.
        
        This creates a CSV that can be directly imported into ACS database.
        
        Args:
            results: List of extraction results
            output_path: Path for CSV file
            
        Returns:
            Path to created CSV file
        """
        # Define Import Script compatible columns
        columns = [
            "MPN",
            "Manufacturer", 
            "Category",
            "Description",
            "Vin_min",
            "Vin_max",
            "Vout_min",
            "Vout_max",
            "Iout_max",
            "Iq_typ",
            "Efficiency_typ",
            "Frequency_typ",
            "Package",
            "Status",
            "Datasheet_URL",
            "Notes"
        ]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in results:
                params = result.get("parameters", {})
                metadata = result.get("metadata", {})
                
                row = {
                    "MPN": self._extract_mpn(metadata, params),
                    "Manufacturer": metadata.get("manufacturer", ""),
                    "Category": self._infer_category(params),
                    "Description": "",
                    "Vin_min": params.get("input_voltage_range_min", ""),
                    "Vin_max": params.get("input_voltage_range_max", ""),
                    "Vout_min": params.get("output_voltage_min", ""),
                    "Vout_max": params.get("output_voltage_max", ""),
                    "Iout_max": params.get("output_current_max", ""),
                    "Iq_typ": params.get("quiescent_current_typ", ""),
                    "Efficiency_typ": params.get("efficiency_typ", ""),
                    "Frequency_typ": params.get("switching_frequency_typ", ""),
                    "Package": "",
                    "Status": "Active",
                    "Datasheet_URL": "",
                    "Notes": f"Extracted by Datasheet-ETL v0.4 (confidence: {result.get('verification', {}).get('confidence', 0):.0%})"
                }
                
                writer.writerow(row)
        
        logger.info(f"Exported {len(results)} results for Import Script to {output_path}")
        return output_path
    
    def _extract_mpn(self, metadata: Dict, params: Dict) -> str:
        """Extract MPN from metadata or filename."""
        if "mpn" in metadata:
            return metadata["mpn"]
        
        source = metadata.get("source_file", "")
        if source:
            # Remove extension and common suffixes
            mpn = Path(source).stem
            for suffix in ["_v04", "_extracted", "_datasheet"]:
                mpn = mpn.replace(suffix, "")
            return mpn.upper()
        
        return ""
    
    def _infer_category(self, params: Dict) -> str:
        """Infer component category from parameters."""
        # Check for switching frequency (indicates switching regulator)
        if "switching_frequency_typ" in params:
            return "Buck Converter"
        
        # Check for efficiency (indicates regulator)
        if "efficiency_typ" in params:
            return "DC-DC Converter"
        
        # Check for quiescent current (indicates LDO)
        if "quiescent_current_typ" in params:
            iq = params.get("quiescent_current_typ", 0)
            if isinstance(iq, (int, float)) and iq < 100:
                return "LDO"
        
        return "Unknown"


# ============================================================================
# Summary Report Generator
# ============================================================================

def generate_batch_report(batch_result: BatchResult, output_path: str) -> str:
    """
    Generate a summary report for batch processing.
    
    Args:
        batch_result: BatchResult from batch processing
        output_path: Path for report file
        
    Returns:
        Path to created report file
    """
    report = f"""# Datasheet ETL Batch Processing Report

**Generated:** {datetime.now().isoformat()}
**Duration:** {batch_result.duration_seconds:.1f} seconds

## Summary

| Metric | Value |
|--------|-------|
| Total Files | {batch_result.total} |
| Success | {batch_result.success} |
| Failed | {batch_result.failed} |
| Skipped | {batch_result.skipped} |
| Success Rate | {batch_result.success / max(batch_result.total - batch_result.skipped, 1):.1%} |

## Successful Extractions

| File | Confidence | Output |
|------|------------|--------|
"""
    
    for r in batch_result.results:
        report += f"| {r['file']} | {r['confidence']:.0%} | {r['output']} |\n"
    
    if batch_result.errors:
        report += "\n## Errors\n\n"
        report += "| File | Error |\n|------|-------|\n"
        for e in batch_result.errors:
            report += f"| {e['file']} | {e['error']} |\n"
    
    report += "\n---\n*Generated by Datasheet-ETL v0.4*\n"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Generated batch report: {output_path}")
    return output_path


# ============================================================================
# CLI for Batch Processing
# ============================================================================

def main():
    """CLI entry point for batch processing."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    if len(sys.argv) < 3:
        print("Usage: python batch_export.py <input_dir> <output_dir> [--csv]")
        print("\nExamples:")
        print("  python batch_export.py datasheets/ output/")
        print("  python batch_export.py datasheets/ output/ --csv")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    export_csv = "--csv" in sys.argv
    
    # Import pipeline
    from pipeline import DatasheetPipeline
    
    # Initialize
    pipeline = DatasheetPipeline()
    processor = BatchProcessor(
        pipeline,
        on_progress=lambda cur, tot, name: print(f"[{cur}/{tot}] {name}")
    )
    
    # Process
    print(f"\nProcessing PDFs in: {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    result = processor.process_directory(input_dir, output_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, "batch_report.md")
    generate_batch_report(result, report_path)
    
    # Export CSV if requested
    if export_csv and result.results:
        # Load all results
        all_results = []
        for r in result.results:
            with open(r["output"], 'r') as f:
                all_results.append(json.load(f))
        
        exporter = Exporter()
        csv_path = os.path.join(output_dir, "extracted_components.csv")
        exporter.export_for_import_script(all_results, csv_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total: {result.total}")
    print(f"Success: {result.success}")
    print(f"Failed: {result.failed}")
    print(f"Skipped: {result.skipped}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Report: {report_path}")
    
    if export_csv:
        print(f"CSV Export: {csv_path}")


if __name__ == "__main__":
    main()
