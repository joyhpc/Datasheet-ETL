# Datasheet Table Extraction: Edge Cases & Challenges

> **Author:** Axiom
> **Date:** 2026-02-13
> **Purpose:** Deep dive into irregular table extraction challenges for discussion with Sirius

## 1. The Core Problem

Semiconductor datasheets are NOT designed for machine parsing. They are designed for human readability, which means:
- Visual cues (bold, color, spacing) convey meaning
- Tables often break PDF parsing assumptions
- Context is scattered across pages

## 2. Irregular Table Categories

### 2.1 Merged Cells (合并单元格)
**Example:**
```
| Parameter          | Conditions      | Min | Typ | Max | Unit |
|--------------------|-----------------|-----|-----|-----|------|
| VIN Operating      |                 | 4.2 |     | 36  | V    |
| Range              |                 |     |     |     |      |  ← Row continuation!
| UVLO Threshold     | Rising          | 3.8 | 4.1 | 4.4 | V    |
|                    | Falling         | 3.5 | 3.8 | 4.1 | V    |  ← Merged "UVLO Threshold"
```

**Challenge:** pdfplumber may return these as separate rows, losing the semantic connection.

**Solution Strategy:**
1. Detect empty first-column cells → merge with previous row
2. Use Vision Model to validate merged cell detection
3. Heuristic: If row[0] is empty and row has values, it's a continuation

### 2.2 Invisible Borders (隐形边框)
**Example:** Some TI datasheets use whitespace alignment instead of lines.

```
Parameter              Min    Typ    Max    Unit
VIN Operating Range    4.2           36     V
Quiescent Current             25     40     µA
```

**Challenge:** pdfplumber's `extract_tables()` relies on line detection. No lines = no table detected.

**Solution Strategy:**
1. Fall back to text-based extraction using character positions
2. Detect column alignment by analyzing x-coordinates of text
3. Use Vision Model as primary extractor for borderless tables

### 2.3 Multi-line Headers (多行表头)
**Example:**
```
|           | Test      |        Output Voltage        |
| Parameter | Condition | Min    | Typ    | Max    | Unit |
```

**Challenge:** Two-row headers are common. First row often contains category groupings.

**Solution Strategy:**
1. Detect if first row has fewer cells than second row
2. Merge header rows: "Output Voltage - Min", "Output Voltage - Typ"
3. Use keyword detection: "Parameter", "Min", "Typ", "Max" usually in final header row

### 2.4 Cross-page Tables (跨页表格)
**Example:** Electrical Characteristics often spans 2-3 pages.

**Challenge:** Each page is processed independently. Table continuity is lost.

**Solution Strategy:**
1. Detect "continued" or "(continued)" in page text
2. If table on page N has same column structure as page N-1, merge them
3. Store table state across pages during extraction

### 2.5 Nested Conditions & Footnotes (嵌套条件)
**Example:**
```
| Parameter | Conditions           | Min | Typ | Max | Unit |
|-----------|----------------------|-----|-----|-----|------|
| IQ        | VFB = 1.1V (Note 1)  |     | 25  | 40  | µA   |

Note 1: Measured at VIN = 12V, TA = 25°C
```

**Challenge:** Conditions are embedded in cells, footnotes are at page bottom.

**Solution Strategy:**
1. Extract footnote markers (Note 1, *, †) from cells
2. Parse footnotes section at page bottom
3. Link footnotes to cells in output JSON

## 3. Proposed Test Cases

| Test ID | Description | Expected Behavior |
|---------|-------------|-------------------|
| TC-001 | Simple table with visible borders | 100% accuracy |
| TC-002 | Table with merged cells (vertical) | Detect and merge rows |
| TC-003 | Table with merged cells (horizontal) | Detect and merge columns |
| TC-004 | Borderless table (whitespace-aligned) | Fall back to Vision Model |
| TC-005 | Two-row header | Merge headers correctly |
| TC-006 | Cross-page table | Detect continuation, merge |
| TC-007 | Table with footnotes | Extract and link footnotes |
| TC-008 | Mixed: merged + footnotes + borderless | Graceful degradation |

## 4. Validation Strategy

### 4.1 Ground Truth Creation
1. Manually annotate 5 representative datasheets
2. Create JSON "expected output" for each table
3. Compare extraction results against ground truth

### 4.2 Confidence Scoring
Each extracted table should have:
- `confidence`: 0.0 - 1.0 overall score
- `flags`: List of detected issues ["merged_cells", "borderless", "footnotes"]
- `extraction_method`: "rule_based" | "vision_model" | "hybrid"

### 4.3 Human-in-the-Loop
For low-confidence extractions (< 0.7):
1. Flag for human review
2. Store original PDF region as image
3. Allow manual correction → feed back to training

## 5. Questions for Sirius

1. **Priority:** Which edge case should we tackle first for MVP?
   - Recommendation: Merged cells (most common in Electrical Characteristics)

2. **Vision Model Budget:** What's acceptable cost per datasheet?
   - Current estimate: $0.50 for 30 pages with Vision fallback

3. **Accuracy Target:** What's minimum acceptable accuracy for MVP?
   - Recommendation: 90% for simple tables, 70% for complex tables

4. **Integration:** How should extracted data flow into HPC_RESO?
   - Option A: Direct database insert
   - Option B: JSON file → manual review → import

---

*This document is a living analysis. Updates will be made as we discover more edge cases.*
