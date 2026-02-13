# Datasheet ETL: Technical Implementation Plan v0.2

> **Author:** Axiom
> **Date:** 2026-02-13
> **Status:** Draft → Ready for Review

## 1. Executive Summary
This document outlines the concrete implementation steps for extracting structured data from semiconductor datasheets (PDFs). The focus is on **tables** (Electrical Characteristics) and **diagrams** (Block Diagrams, Typical Applications).

## 2. Target Datasheet Selection
For initial validation, we will use:
-   **Primary:** TI LMR51430 (Simple Synchronous Buck Converter, ~30 pages)
    -   Why: Clean layout, representative tables, typical block diagram.
    -   Download: https://www.ti.com/lit/ds/symlink/lmr51430.pdf
-   **Secondary:** TI TS5A3159 (Analog Switch, ~20 pages)
    -   Why: Simpler, good for testing table extraction.

## 3. Implementation Phases

### Phase 1: PDF Ingestion & Page Splitting (Week 1)
**Goal:** Load PDF, split into pages, identify regions of interest.

**Tools:**
-   `pdfplumber` (Python): For text/table extraction with coordinates.
-   `PyMuPDF` (fitz): For image extraction and rendering.

**Code Skeleton:**
```python
import pdfplumber
from pathlib import Path

def ingest_pdf(pdf_path: str):
    """Load PDF and yield page objects with metadata."""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            yield {
                "page_num": i + 1,
                "width": page.width,
                "height": page.height,
                "text": page.extract_text(),
                "tables": page.extract_tables(),
                "chars": page.chars,  # For layout analysis
            }
```

### Phase 2: Table Extraction (Week 1-2)
**Goal:** Extract Electrical Characteristics tables with proper structure.

**Challenge:** Merged cells, multi-line headers, invisible borders.

**Strategy (Hybrid):**
1.  **Rule-Based (Fast Path):**
    -   Use `pdfplumber.extract_tables()` for clean grids.
    -   Detect header row by keywords: "Parameter", "Min", "Typ", "Max", "Unit".
    -   Normalize to JSON schema.

2.  **Vision-Assisted (Slow Path):**
    -   If table detection fails or confidence < 80%, crop table region as image.
    -   Send to Vision Model with prompt:
        ```
        Extract this table from a semiconductor datasheet.
        Return JSON with columns: Parameter, Test_Conditions, Min, Typ, Max, Unit.
        Handle merged cells and multi-line text.
        ```

**Output Schema:**
```json
{
  "table_id": "elec_char_1",
  "page": 5,
  "title": "Electrical Characteristics",
  "columns": ["Parameter", "Test_Conditions", "Min", "Typ", "Max", "Unit"],
  "rows": [
    {"Parameter": "VIN_UVLO", "Test_Conditions": "Rising", "Min": "3.8", "Typ": "4.1", "Max": "4.4", "Unit": "V"},
    ...
  ],
  "confidence": 0.95,
  "extraction_method": "rule_based"
}
```

### Phase 3: Diagram Understanding (Week 2-3)
**Goal:** Extract topology from block diagrams.

**Challenge:** This is fundamentally a computer vision + reasoning problem.

**Strategy:**
1.  **Region Detection:**
    -   Identify diagram regions by detecting large image areas or specific captions ("Figure 1", "Block Diagram").
    -   Use heuristics: Large bounding boxes with few text chars inside.

2.  **Vision Model Extraction:**
    -   Crop diagram region.
    -   Send to Vision Model with prompt:
        ```
        This is a functional block diagram from a DC/DC converter datasheet.
        Extract the signal flow as a graph:
        - Nodes: Component blocks (e.g., "Error Amplifier", "PWM Comparator", "Driver")
        - Edges: Connections with labels (e.g., "FB -> Error Amplifier")
        Return as JSON: {"nodes": [...], "edges": [...]}
        ```

**Output Schema:**
```json
{
  "diagram_id": "block_diagram_1",
  "page": 1,
  "caption": "Functional Block Diagram",
  "nodes": [
    {"id": "n1", "label": "Error Amplifier", "type": "block"},
    {"id": "n2", "label": "PWM Comparator", "type": "block"},
    ...
  ],
  "edges": [
    {"source": "FB", "target": "n1", "label": "Feedback"},
    {"source": "n1", "target": "n2", "label": "Error Signal"},
    ...
  ]
}
```

### Phase 4: Integration & Validation (Week 3)
**Goal:** End-to-end pipeline test.

**Steps:**
1.  Run pipeline on LMR51430 datasheet.
2.  Compare extracted data against manually annotated "ground truth".
3.  Calculate accuracy metrics (Precision, Recall, F1).
4.  Iterate on prompts and heuristics.

## 4. API Cost Estimation
| Operation | Model | Est. Cost/Page |
|-----------|-------|----------------|
| Table Extraction (Vision) | GPT-4o-Mini | ~$0.01 |
| Diagram Understanding | GPT-4o | ~$0.05 |
| Full Datasheet (30 pages) | Mixed | ~$0.50 |

## 5. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Vision Model hallucination | Cross-validate with rule-based extraction |
| High API cost at scale | Cache results, batch processing |
| Complex table layouts | Iterative prompt engineering |

## 6. Next Steps (Immediate)
1.  [ ] Download LMR51430 datasheet to `datasheets/` folder.
2.  [ ] Implement `ingest_pdf()` function.
3.  [ ] Test `pdfplumber` table extraction on Page 5 (Electrical Characteristics).
4.  [ ] Design Vision Model prompt for table fallback.

---
*This plan follows First Principles: Start with the physics of the problem (PDF structure), then layer intelligence (Vision AI) only where rules fail.*
