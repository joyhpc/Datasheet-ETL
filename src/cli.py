#!/usr/bin/env python3
"""
Datasheet ETL: Command Line Interface

A unified CLI for all Datasheet ETL operations.

Usage:
    python cli.py extract <pdf_path> [--output <dir>]
    python cli.py batch <input_dir> <output_dir> [--csv]
    python cli.py verify <json_path>
    python cli.py export <json_path> --format csv
    python cli.py stats

Author: Axiom
Date: 2026-02-14
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_extract(args):
    """Extract data from a single datasheet."""
    from pipeline import DatasheetPipeline, PipelineConfig
    
    config = PipelineConfig(
        vision_model=args.model or "gpt-4o-mini",
        include_conflicts=True
    )
    
    pipeline = DatasheetPipeline(config)
    
    output_dir = args.output or "output"
    pdf_name = Path(args.pdf_path).stem
    output_path = f"{output_dir}/{pdf_name}_v04.json"
    
    print(f"\n📄 Extracting: {args.pdf_path}")
    print(f"📁 Output: {output_path}\n")
    
    result = pipeline.process(args.pdf_path, output_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Status: {result['verification']['status']}")
    print(f"Confidence: {result['verification']['confidence']:.0%}")
    print(f"Parameters: {len(result.get('parameters', {}))}")
    
    if args.verbose:
        print("\nParameters:")
        for k, v in result.get('parameters', {}).items():
            print(f"  {k}: {v}")
    
    return 0


def cmd_batch(args):
    """Batch process multiple datasheets."""
    from pipeline import DatasheetPipeline
    from batch_export import BatchProcessor, Exporter, generate_batch_report
    
    pipeline = DatasheetPipeline()
    processor = BatchProcessor(
        pipeline,
        on_progress=lambda cur, tot, name: print(f"[{cur}/{tot}] {name}")
    )
    
    print(f"\n📂 Input: {args.input_dir}")
    print(f"📁 Output: {args.output_dir}\n")
    
    result = processor.process_directory(args.input_dir, args.output_dir)
    
    # Generate report
    report_path = f"{args.output_dir}/batch_report.md"
    generate_batch_report(result, report_path)
    
    # Export CSV if requested
    if args.csv and result.results:
        all_results = []
        for r in result.results:
            with open(r["output"], 'r') as f:
                all_results.append(json.load(f))
        
        exporter = Exporter()
        csv_path = f"{args.output_dir}/extracted_components.csv"
        exporter.export_for_import_script(all_results, csv_path)
        print(f"\n📊 CSV Export: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ BATCH COMPLETE")
    print("=" * 50)
    print(f"Total: {result.total}")
    print(f"Success: {result.success}")
    print(f"Failed: {result.failed}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    
    return 0 if result.failed == 0 else 1


def cmd_verify(args):
    """Verify an extraction result."""
    from double_verifier import DoubleVerifier, VerificationStatus
    
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n🔍 Verifying: {args.json_path}\n")
    
    verification = data.get('verification', {})
    
    print(f"Status: {verification.get('status', 'unknown')}")
    print(f"Confidence: {verification.get('confidence', 0):.0%}")
    print(f"Resolution: {verification.get('resolution_method', 'unknown')}")
    print(f"Conflicts: {verification.get('conflict_count', 0)}")
    
    if 'conflicts' in data:
        print("\nConflicts:")
        for c in data['conflicts']:
            print(f"  - {c['field']}: '{c['vision_value']}' vs '{c['rule_value']}' -> {c['resolution']}")
    
    return 0


def cmd_export(args):
    """Export extraction results to different formats."""
    from batch_export import Exporter
    
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    if args.format == "csv":
        exporter = Exporter()
        output_path = args.output or args.json_path.replace('.json', '.csv')
        exporter.export_for_import_script([data], output_path)
        print(f"✅ Exported to: {output_path}")
    elif args.format == "review":
        from review_adapter import convert_file
        output_path = args.output or args.json_path.replace('.json', '_review.json')
        convert_file(args.json_path, output_path)
        print(f"✅ Exported Review UI format to: {output_path}")
    else:
        print(f"❌ Unknown format: {args.format}")
        return 1
    
    return 0


def cmd_stats(args):
    """Show pipeline statistics."""
    from pipeline import DatasheetPipeline
    
    pipeline = DatasheetPipeline()
    stats = pipeline.get_stats()
    
    print("\n📊 Pipeline Statistics")
    print("=" * 50)
    print(f"Processed: {stats['processed_count']}")
    print(f"Review Queue: {stats['review_queue_size']}")
    print(f"\nVision Stats:")
    print(f"  Model: {stats['vision_stats']['model']}")
    print(f"  Calls: {stats['vision_stats']['calls']}")
    print(f"  Est. Cost: ${stats['vision_stats']['estimated_cost']:.4f}")
    print(f"\nVerifier Stats:")
    for k, v in stats['verifier_stats'].items():
        print(f"  {k}: {v}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Datasheet ETL - Extract structured data from semiconductor datasheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s extract datasheets/lmr51430.pdf
  %(prog)s batch datasheets/ output/ --csv
  %(prog)s verify output/lmr51430_v04.json
  %(prog)s export output/lmr51430_v04.json --format csv
  %(prog)s stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract from single PDF')
    extract_parser.add_argument('pdf_path', help='Path to PDF file')
    extract_parser.add_argument('--output', '-o', help='Output directory')
    extract_parser.add_argument('--model', '-m', help='Vision model to use')
    extract_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process directory')
    batch_parser.add_argument('input_dir', help='Input directory with PDFs')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--csv', action='store_true', help='Export CSV for Import Script')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify extraction result')
    verify_parser.add_argument('json_path', help='Path to JSON result file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export to different format')
    export_parser.add_argument('json_path', help='Path to JSON result file')
    export_parser.add_argument('--format', '-f', default='csv', help='Export format (csv, review)')
    export_parser.add_argument('--output', '-o', help='Output file path')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show pipeline statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to command handler
    commands = {
        'extract': cmd_extract,
        'batch': cmd_batch,
        'verify': cmd_verify,
        'export': cmd_export,
        'stats': cmd_stats,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
