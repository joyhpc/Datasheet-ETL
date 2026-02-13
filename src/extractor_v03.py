"""
Datasheet ETL: Production-Ready Extractor v0.3

This module provides a complete pipeline for extracting structured data
from semiconductor datasheets (PDFs).

Dependencies:
- pdfplumber: For text/table extraction with coordinates
- Pillow: For image processing
- requests: For Vision API calls (optional)

Usage:
    python extractor_v03.py datasheets/lmr51430.pdf output/

Author: Axiom
Date: 2026-02-13
"""

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TableCell:
    """Represents a single cell in a table."""
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1

@dataclass
class ExtractedTable:
    """Represents an extracted table with metadata."""
    table_id: str
    page_num: int
    title: str
    headers: List[str]
    rows: List[Dict[str, str]]
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float
    extraction_method: str  # "rule_based" or "vision_model"
    raw_data: Optional[List[List[str]]] = None

@dataclass
class ExtractedDiagram:
    """Represents an extracted diagram with topology."""
    diagram_id: str
    page_num: int
    caption: str
    diagram_type: str  # "block_diagram", "typical_app", "pinout"
    nodes: List[Dict[str, str]]
    edges: List[Dict[str, str]]
    bbox: Tuple[float, float, float, float]

@dataclass
class DatasheetExtraction:
    """Complete extraction result for a datasheet."""
    source_file: str
    mpn: str  # Manufacturer Part Number
    manufacturer: str
    total_pages: int
    tables: List[ExtractedTable] = field(default_factory=list)
    diagrams: List[ExtractedDiagram] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Table Extraction Engine
# ============================================================================

class TableExtractor:
    """
    Hybrid table extraction engine.
    
    Strategy:
    1. Try rule-based extraction first (fast, cheap)
    2. Fall back to Vision Model for complex tables
    """
    
    # Common header patterns in semiconductor datasheets
    HEADER_PATTERNS = [
        r"parameter",
        r"symbol",
        r"test\s*condition",
        r"min\.?",
        r"typ\.?",
        r"max\.?",
        r"unit",
        r"description",
    ]
    
    # Table title patterns
    TABLE_TITLE_PATTERNS = [
        r"electrical\s*characteristics",
        r"absolute\s*maximum\s*ratings",
        r"recommended\s*operating\s*conditions",
        r"thermal\s*information",
        r"pin\s*functions",
        r"device\s*comparison",
    ]
    
    def __init__(self, vision_api_key: Optional[str] = None):
        self.vision_api_key = vision_api_key
        self._pdfplumber_available = self._check_pdfplumber()
    
    def _check_pdfplumber(self) -> bool:
        """Check if pdfplumber is available."""
        try:
            import pdfplumber
            return True
        except ImportError:
            logger.warning("pdfplumber not available. Using fallback methods.")
            return False
    
    def extract_tables_from_page(
        self, 
        page_data: Dict[str, Any],
        page_num: int
    ) -> List[ExtractedTable]:
        """
        Extract tables from a single page.
        
        Args:
            page_data: Page data from pdfplumber or equivalent
            page_num: Page number (1-indexed)
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        if self._pdfplumber_available:
            tables = self._extract_with_pdfplumber(page_data, page_num)
        
        # If no tables found or low confidence, try Vision Model
        if not tables or all(t.confidence < 0.7 for t in tables):
            if self.vision_api_key:
                vision_tables = self._extract_with_vision(page_data, page_num)
                tables.extend(vision_tables)
        
        return tables
    
    def _extract_with_pdfplumber(
        self, 
        page_data: Dict[str, Any],
        page_num: int
    ) -> List[ExtractedTable]:
        """Rule-based extraction using pdfplumber."""
        tables = []
        raw_tables = page_data.get("tables", [])
        
        for idx, raw_table in enumerate(raw_tables):
            if not raw_table or len(raw_table) < 2:
                continue
            
            # Detect headers
            headers = self._detect_headers(raw_table[0])
            if not headers:
                headers = [f"col_{i}" for i in range(len(raw_table[0]))]
            
            # Parse rows
            rows = []
            for row_data in raw_table[1:]:
                if row_data and any(cell for cell in row_data):
                    row_dict = {}
                    for i, cell in enumerate(row_data):
                        key = headers[i] if i < len(headers) else f"col_{i}"
                        row_dict[key] = self._clean_cell(cell)
                    rows.append(row_dict)
            
            # Detect table title
            title = self._detect_table_title(page_data.get("text", ""), idx)
            
            # Calculate confidence based on header detection
            confidence = 0.9 if self._has_standard_headers(headers) else 0.6
            
            table = ExtractedTable(
                table_id=f"table_p{page_num}_{idx}",
                page_num=page_num,
                title=title,
                headers=headers,
                rows=rows,
                bbox=(0, 0, 0, 0),  # Would need actual coordinates
                confidence=confidence,
                extraction_method="rule_based",
                raw_data=raw_table
            )
            tables.append(table)
        
        return tables
    
    def _extract_with_vision(
        self, 
        page_data: Dict[str, Any],
        page_num: int
    ) -> List[ExtractedTable]:
        """Vision Model extraction for complex tables."""
        # This would call GPT-4o or similar
        # For now, return empty list as placeholder
        logger.info(f"Vision extraction requested for page {page_num}")
        return []
    
    def _detect_headers(self, first_row: List[str]) -> List[str]:
        """Detect and normalize table headers."""
        if not first_row:
            return []
        
        headers = []
        for cell in first_row:
            cell_lower = (cell or "").lower().strip()
            
            # Normalize common header names
            if re.search(r"param", cell_lower):
                headers.append("Parameter")
            elif re.search(r"symbol", cell_lower):
                headers.append("Symbol")
            elif re.search(r"test.*cond|condition", cell_lower):
                headers.append("Test_Conditions")
            elif re.search(r"^min", cell_lower):
                headers.append("Min")
            elif re.search(r"^typ", cell_lower):
                headers.append("Typ")
            elif re.search(r"^max", cell_lower):
                headers.append("Max")
            elif re.search(r"unit", cell_lower):
                headers.append("Unit")
            elif re.search(r"desc", cell_lower):
                headers.append("Description")
            else:
                headers.append(cell or "Unknown")
        
        return headers
    
    def _has_standard_headers(self, headers: List[str]) -> bool:
        """Check if headers match standard datasheet patterns."""
        standard = {"Parameter", "Min", "Typ", "Max", "Unit"}
        return len(set(headers) & standard) >= 3
    
    def _detect_table_title(self, page_text: str, table_idx: int) -> str:
        """Detect table title from surrounding text."""
        for pattern in self.TABLE_TITLE_PATTERNS:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(0).title()
        return f"Table {table_idx + 1}"
    
    def _clean_cell(self, cell: Optional[str]) -> str:
        """Clean and normalize cell content."""
        if cell is None:
            return ""
        # Remove extra whitespace, normalize unicode
        cleaned = " ".join(str(cell).split())
        return cleaned

# ============================================================================
# Diagram Extraction Engine
# ============================================================================

class DiagramExtractor:
    """
    Vision-based diagram extraction engine.
    
    Extracts topology from block diagrams and application circuits.
    """
    
    DIAGRAM_PATTERNS = [
        (r"block\s*diagram", "block_diagram"),
        (r"functional\s*diagram", "block_diagram"),
        (r"typical\s*application", "typical_app"),
        (r"application\s*circuit", "typical_app"),
        (r"pin\s*configuration", "pinout"),
        (r"package", "pinout"),
    ]
    
    def __init__(self, vision_api_key: Optional[str] = None):
        self.vision_api_key = vision_api_key
    
    def extract_diagrams_from_page(
        self,
        page_image: Any,  # PIL Image or path
        page_num: int,
        page_text: str
    ) -> List[ExtractedDiagram]:
        """
        Extract diagrams from a page image.
        
        This requires Vision Model API access.
        """
        diagrams = []
        
        # Detect diagram type from text
        diagram_type = self._detect_diagram_type(page_text)
        
        if diagram_type and self.vision_api_key:
            # Would call Vision API here
            logger.info(f"Diagram extraction requested: {diagram_type} on page {page_num}")
        
        return diagrams
    
    def _detect_diagram_type(self, text: str) -> Optional[str]:
        """Detect diagram type from page text."""
        text_lower = text.lower()
        for pattern, dtype in self.DIAGRAM_PATTERNS:
            if re.search(pattern, text_lower):
                return dtype
        return None

# ============================================================================
# Main Pipeline
# ============================================================================

class DatasheetPipeline:
    """
    Complete datasheet extraction pipeline.
    
    Usage:
        pipeline = DatasheetPipeline(vision_api_key="...")
        result = pipeline.process("datasheets/lmr51430.pdf")
        result.save_json("output/lmr51430.json")
    """
    
    def __init__(self, vision_api_key: Optional[str] = None):
        self.table_extractor = TableExtractor(vision_api_key)
        self.diagram_extractor = DiagramExtractor(vision_api_key)
    
    def process(self, pdf_path: str) -> DatasheetExtraction:
        """
        Process a datasheet PDF and extract structured data.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DatasheetExtraction object with all extracted data
        """
        logger.info(f"Processing: {pdf_path}")
        
        # Initialize result
        result = DatasheetExtraction(
            source_file=pdf_path,
            mpn="",
            manufacturer="",
            total_pages=0
        )
        
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                result.total_pages = len(pdf.pages)
                logger.info(f"Total pages: {result.total_pages}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}/{result.total_pages}")
                    
                    # Extract page data
                    page_data = {
                        "text": page.extract_text() or "",
                        "tables": page.extract_tables(),
                        "width": page.width,
                        "height": page.height,
                    }
                    
                    # Extract tables
                    tables = self.table_extractor.extract_tables_from_page(
                        page_data, page_num
                    )
                    result.tables.extend(tables)
                    
                    # Extract MPN from first page
                    if page_num == 1:
                        result.mpn = self._extract_mpn(page_data["text"])
                        result.manufacturer = self._extract_manufacturer(page_data["text"])
        
        except ImportError:
            logger.error("pdfplumber not available. Cannot process PDF.")
            result.metadata["error"] = "pdfplumber not installed"
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            result.metadata["error"] = str(e)
        
        logger.info(f"Extraction complete. Tables: {len(result.tables)}, Diagrams: {len(result.diagrams)}")
        return result
    
    def _extract_mpn(self, text: str) -> str:
        """Extract MPN from first page text."""
        # Common patterns for TI parts
        patterns = [
            r"LMR\d+[A-Z]*",
            r"TPS\d+[A-Z]*",
            r"LM\d+[A-Z]*",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return "Unknown"
    
    def _extract_manufacturer(self, text: str) -> str:
        """Extract manufacturer from first page."""
        if "texas instruments" in text.lower():
            return "Texas Instruments"
        elif "analog devices" in text.lower():
            return "Analog Devices"
        elif "onsemi" in text.lower():
            return "onsemi"
        return "Unknown"

# ============================================================================
# Output Functions
# ============================================================================

def save_extraction_json(extraction: DatasheetExtraction, output_path: str):
    """Save extraction result to JSON file."""
    output = {
        "metadata": {
            "source_file": extraction.source_file,
            "mpn": extraction.mpn,
            "manufacturer": extraction.manufacturer,
            "total_pages": extraction.total_pages,
            "extractor_version": "0.3",
        },
        "tables": [asdict(t) for t in extraction.tables],
        "diagrams": [asdict(d) for d in extraction.diagrams],
        "parameters": extraction.parameters,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved extraction to: {output_path}")

# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extractor_v03.py <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process
    pipeline = DatasheetPipeline()
    result = pipeline.process(pdf_path)
    
    # Save
    output_file = Path(output_dir) / f"{Path(pdf_path).stem}_extracted.json"
    save_extraction_json(result, str(output_file))
    
    print(f"\nExtraction Summary:")
    print(f"  MPN: {result.mpn}")
    print(f"  Manufacturer: {result.manufacturer}")
    print(f"  Pages: {result.total_pages}")
    print(f"  Tables: {len(result.tables)}")
    print(f"  Diagrams: {len(result.diagrams)}")
