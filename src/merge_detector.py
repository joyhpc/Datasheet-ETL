"""
Datasheet ETL: Merged Cell Detection Module

This module handles detection and resolution of merged cells in datasheet tables.
Focus: Horizontal merges (column spanning) as per Sirius decision.

Key Insight: Merged cells in datasheets often indicate:
1. Category headers spanning multiple columns
2. Grouped parameters (e.g., "Output Voltage" spanning Min/Typ/Max)
3. Condition blocks (e.g., test conditions applying to multiple rows)

Author: Axiom
Date: 2026-02-13
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MergedCell:
    """Represents a detected merged cell."""
    row: int
    col_start: int
    col_end: int
    content: str
    merge_type: str  # "horizontal" or "vertical"
    confidence: float

@dataclass
class TableWithMerges:
    """Table data with merge information."""
    headers: List[str]
    rows: List[List[str]]
    merges: List[MergedCell]
    resolved_rows: List[Dict[str, str]]  # After merge resolution

class MergedCellDetector:
    """
    Detects and resolves merged cells in extracted tables.
    
    Strategy (per Sirius decision):
    1. Horizontal merges first (column spanning)
    2. 60% confidence threshold for Vision fallback
    """
    
    CONFIDENCE_THRESHOLD = 0.60  # Per Sirius decision
    
    def __init__(self):
        self.detected_merges: List[MergedCell] = []
    
    def detect_horizontal_merges(
        self,
        raw_table: List[List[str]],
        char_positions: Optional[List[List[Tuple[float, float]]]] = None
    ) -> List[MergedCell]:
        """
        Detect horizontal merges (cells spanning multiple columns).
        
        Heuristics:
        1. Cell content appears once but spans multiple column positions
        2. Adjacent cells are empty where content should be
        3. Header row has fewer cells than data rows
        
        Args:
            raw_table: 2D list of cell contents
            char_positions: Optional x-coordinates for each cell
            
        Returns:
            List of detected MergedCell objects
        """
        merges = []
        
        if not raw_table or len(raw_table) < 2:
            return merges
        
        # Method 1: Header row analysis
        header_merges = self._detect_header_merges(raw_table)
        merges.extend(header_merges)
        
        # Method 2: Empty cell pattern analysis
        pattern_merges = self._detect_empty_cell_patterns(raw_table)
        merges.extend(pattern_merges)
        
        # Method 3: Position-based detection (if coordinates available)
        if char_positions:
            position_merges = self._detect_by_positions(raw_table, char_positions)
            merges.extend(position_merges)
        
        # Deduplicate and filter by confidence
        merges = self._deduplicate_merges(merges)
        self.detected_merges = merges
        
        return merges
    
    def _detect_header_merges(self, raw_table: List[List[str]]) -> List[MergedCell]:
        """
        Detect merges in header rows.
        
        Common pattern in datasheets:
        Row 0: |           | Output Voltage |           |      |
        Row 1: | Parameter | Min | Typ | Max | Unit |
        
        "Output Voltage" spans Min/Typ/Max columns.
        """
        merges = []
        
        if len(raw_table) < 2:
            return merges
        
        header_row = raw_table[0]
        data_row = raw_table[1]
        
        # Check if header has fewer non-empty cells than data row
        header_cells = [c for c in header_row if c and c.strip()]
        data_cells = [c for c in data_row if c and c.strip()]
        
        if len(header_cells) < len(data_cells):
            # Likely has merged headers
            logger.info(f"Header merge detected: {len(header_cells)} headers for {len(data_cells)} columns")
            
            # Find which columns are spanned
            col_idx = 0
            for i, cell in enumerate(header_row):
                if cell and cell.strip():
                    # Count how many empty cells follow
                    span = 1
                    for j in range(i + 1, len(header_row)):
                        if not header_row[j] or not header_row[j].strip():
                            span += 1
                        else:
                            break
                    
                    if span > 1:
                        merges.append(MergedCell(
                            row=0,
                            col_start=i,
                            col_end=i + span - 1,
                            content=cell.strip(),
                            merge_type="horizontal",
                            confidence=0.85
                        ))
        
        return merges
    
    def _detect_empty_cell_patterns(self, raw_table: List[List[str]]) -> List[MergedCell]:
        """
        Detect merges by analyzing empty cell patterns.
        
        Pattern: Content in cell followed by empty cells in same row
        suggests horizontal merge.
        """
        merges = []
        
        for row_idx, row in enumerate(raw_table):
            i = 0
            while i < len(row):
                cell = row[i]
                if cell and cell.strip():
                    # Check for trailing empty cells
                    span = 1
                    for j in range(i + 1, len(row)):
                        if not row[j] or not row[j].strip():
                            # Check if this is truly empty or just missing data
                            # Heuristic: If multiple consecutive empties, likely merge
                            span += 1
                        else:
                            break
                    
                    # Only consider as merge if span > 1 and pattern is consistent
                    if span > 1 and span <= 4:  # Reasonable merge size
                        # Additional check: Is this a category header?
                        if self._is_category_header(cell):
                            merges.append(MergedCell(
                                row=row_idx,
                                col_start=i,
                                col_end=i + span - 1,
                                content=cell.strip(),
                                merge_type="horizontal",
                                confidence=0.70
                            ))
                    
                    i += span
                else:
                    i += 1
        
        return merges
    
    def _detect_by_positions(
        self,
        raw_table: List[List[str]],
        char_positions: List[List[Tuple[float, float]]]
    ) -> List[MergedCell]:
        """
        Detect merges using character x-coordinates.
        
        If a cell's content spans a wider x-range than typical,
        it's likely merged.
        """
        merges = []
        
        # Calculate typical column widths
        col_widths = self._calculate_column_widths(char_positions)
        
        for row_idx, (row, positions) in enumerate(zip(raw_table, char_positions)):
            for col_idx, (cell, pos) in enumerate(zip(row, positions)):
                if cell and pos:
                    x_start, x_end = pos
                    cell_width = x_end - x_start
                    
                    # If cell is significantly wider than typical column
                    if col_idx < len(col_widths):
                        typical_width = col_widths[col_idx]
                        if cell_width > typical_width * 1.5:
                            # Estimate how many columns it spans
                            span = int(cell_width / typical_width)
                            if span > 1:
                                merges.append(MergedCell(
                                    row=row_idx,
                                    col_start=col_idx,
                                    col_end=col_idx + span - 1,
                                    content=cell.strip(),
                                    merge_type="horizontal",
                                    confidence=0.90
                                ))
        
        return merges
    
    def _calculate_column_widths(
        self,
        char_positions: List[List[Tuple[float, float]]]
    ) -> List[float]:
        """Calculate typical width for each column."""
        if not char_positions:
            return []
        
        # Find max columns
        max_cols = max(len(row) for row in char_positions)
        widths = [[] for _ in range(max_cols)]
        
        for row in char_positions:
            for col_idx, pos in enumerate(row):
                if pos:
                    x_start, x_end = pos
                    widths[col_idx].append(x_end - x_start)
        
        # Return median width for each column
        return [
            sorted(w)[len(w)//2] if w else 0
            for w in widths
        ]
    
    def _is_category_header(self, text: str) -> bool:
        """
        Check if text looks like a category header.
        
        Category headers in datasheets often:
        - Are in title case or all caps
        - Don't contain numbers (unlike parameter values)
        - Match common patterns
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Common category patterns
        category_patterns = [
            r"^(input|output|power|thermal|timing|electrical)",
            r"characteristics$",
            r"^(absolute|recommended|operating)",
            r"conditions$",
            r"^(dc|ac)\s",
        ]
        
        text_lower = text.lower()
        for pattern in category_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check if mostly alphabetic (not a value)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        return alpha_ratio > 0.8
    
    def _deduplicate_merges(self, merges: List[MergedCell]) -> List[MergedCell]:
        """Remove duplicate merge detections, keeping highest confidence."""
        if not merges:
            return []
        
        # Group by (row, col_start)
        merge_map: Dict[Tuple[int, int], MergedCell] = {}
        
        for merge in merges:
            key = (merge.row, merge.col_start)
            if key not in merge_map or merge.confidence > merge_map[key].confidence:
                merge_map[key] = merge
        
        return list(merge_map.values())
    
    def resolve_merges(
        self,
        raw_table: List[List[str]],
        merges: List[MergedCell]
    ) -> TableWithMerges:
        """
        Resolve detected merges and produce clean table data.
        
        Resolution strategy:
        1. For header merges: Create compound column names
           e.g., "Output Voltage" spanning Min/Typ/Max becomes
           "Output Voltage - Min", "Output Voltage - Typ", etc.
        
        2. For data merges: Propagate value to all spanned columns
        """
        if not raw_table:
            return TableWithMerges([], [], [], [])
        
        # Separate header and data rows
        headers = raw_table[0] if raw_table else []
        data_rows = raw_table[1:] if len(raw_table) > 1 else []
        
        # Resolve header merges
        resolved_headers = self._resolve_header_merges(headers, merges)
        
        # Resolve data row merges
        resolved_rows = []
        for row_idx, row in enumerate(data_rows):
            resolved_row = self._resolve_row_merges(
                row, 
                resolved_headers,
                [m for m in merges if m.row == row_idx + 1]  # +1 for header offset
            )
            resolved_rows.append(resolved_row)
        
        return TableWithMerges(
            headers=resolved_headers,
            rows=data_rows,
            merges=merges,
            resolved_rows=resolved_rows
        )
    
    def _resolve_header_merges(
        self,
        headers: List[str],
        merges: List[MergedCell]
    ) -> List[str]:
        """Resolve header merges into compound names."""
        # For two-row headers, we need special handling
        # The first row contains category headers, second row contains actual column names
        resolved = list(headers)  # Copy
        
        # Find header merges (row 0)
        header_merges = [m for m in merges if m.row == 0]
        
        # If we have header merges, the actual column names are in the data
        # We'll mark which columns belong to which category
        for merge in header_merges:
            category = merge.content
            for col in range(merge.col_start, merge.col_end + 1):
                if col < len(resolved):
                    # Store category info for later use
                    # The actual column name will come from the next row
                    if not resolved[col] or not resolved[col].strip():
                        resolved[col] = category
        
        return resolved
    
    def _resolve_row_merges(
        self,
        row: List[str],
        headers: List[str],
        merges: List[MergedCell]
    ) -> Dict[str, str]:
        """Resolve a data row with merges into a dictionary."""
        result = {}
        
        for col_idx, cell in enumerate(row):
            if col_idx < len(headers):
                header = headers[col_idx]
                
                # Check if this column is part of a merge
                merge = next(
                    (m for m in merges if m.col_start <= col_idx <= m.col_end),
                    None
                )
                
                if merge:
                    # Use merge content for all spanned columns
                    result[header] = merge.content
                else:
                    result[header] = cell.strip() if cell else ""
        
        return result
    
    def get_confidence(self) -> float:
        """
        Get overall confidence for merge detection.
        
        Returns average confidence of all detected merges,
        or 1.0 if no merges detected (simple table).
        """
        if not self.detected_merges:
            return 1.0
        
        return sum(m.confidence for m in self.detected_merges) / len(self.detected_merges)
    
    def needs_vision_fallback(self) -> bool:
        """
        Check if Vision Model fallback is needed.
        
        Per Sirius decision: Use Vision if confidence < 60%
        """
        return self.get_confidence() < self.CONFIDENCE_THRESHOLD


# ============================================================================
# Integration with Extractor
# ============================================================================

def enhance_table_extraction(
    raw_table: List[List[str]],
    page_num: int,
    char_positions: Optional[List[List[Tuple[float, float]]]] = None
) -> Tuple[List[Dict[str, str]], float, bool]:
    """
    Enhanced table extraction with merge detection.
    
    Args:
        raw_table: Raw extracted table data
        page_num: Page number for logging
        char_positions: Optional character positions
        
    Returns:
        Tuple of (resolved_rows, confidence, needs_vision)
    """
    detector = MergedCellDetector()
    
    # Detect merges
    merges = detector.detect_horizontal_merges(raw_table, char_positions)
    
    if merges:
        logger.info(f"Page {page_num}: Detected {len(merges)} horizontal merges")
        for m in merges:
            logger.debug(f"  Merge: row={m.row}, cols={m.col_start}-{m.col_end}, "
                        f"content='{m.content}', confidence={m.confidence:.2f}")
    
    # Resolve merges
    result = detector.resolve_merges(raw_table, merges)
    
    confidence = detector.get_confidence()
    needs_vision = detector.needs_vision_fallback()
    
    if needs_vision:
        logger.warning(f"Page {page_num}: Low confidence ({confidence:.2f}), "
                      f"Vision fallback recommended")
    
    return result.resolved_rows, confidence, needs_vision


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Merged Cell Detector Test")
    print("=" * 50)
    
    # Test case: Table with merged header
    test_table = [
        ["", "Output Voltage", "", "", ""],
        ["Parameter", "Min", "Typ", "Max", "Unit"],
        ["VOUT Accuracy", "-1", "", "+1", "%"],
        ["Load Regulation", "", "0.1", "0.5", "%"],
    ]
    
    print("\nInput Table:")
    for row in test_table:
        print(f"  {row}")
    
    detector = MergedCellDetector()
    merges = detector.detect_horizontal_merges(test_table)
    
    print(f"\nDetected Merges: {len(merges)}")
    for m in merges:
        print(f"  Row {m.row}, Cols {m.col_start}-{m.col_end}: "
              f"'{m.content}' (confidence: {m.confidence:.2f})")
    
    result = detector.resolve_merges(test_table, merges)
    
    print(f"\nResolved Headers: {result.headers}")
    print(f"\nResolved Rows:")
    for row in result.resolved_rows:
        print(f"  {row}")
    
    print(f"\nOverall Confidence: {detector.get_confidence():.2f}")
    print(f"Needs Vision Fallback: {detector.needs_vision_fallback()}")
