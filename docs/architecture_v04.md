# Datasheet ETL: Architecture v0.4

> **Version:** 0.4 (Vision-First + Double Verification)
> **Date:** 2026-02-13
> **Authors:** Axiom (Architecture), Sirius (Implementation)
> **Precision Target:** 99%+ (Mandatory)

## 1. Design Principles

### 1.1 Core Philosophy
- **Vision-First:** Vision Model 是主要提取引擎，不是后备
- **Double Verification:** 每个提取结果必须经过两种方法验证
- **Modular & Extensible:** 每个组件可独立替换、测试、升级
- **Fail-Safe:** 验证失败时标记为 "需人工审核"，不静默通过

### 1.2 Accuracy Tiers
| Tier | Accuracy | Use Case |
|------|----------|----------|
| Gold | 99%+ | Production data, BOM generation |
| Silver | 95%+ | Quick preview, draft selection |
| Bronze | 85%+ | Bulk indexing, search only |

**Default:** Gold tier (99%+)

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Datasheet ETL Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Ingestion  │───▶│  Extraction  │───▶│ Verification │───▶ Output│
│  │    Layer     │    │    Layer     │    │    Layer     │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ PDF Renderer │    │Vision Engine │    │ Rule Checker │           │
│  │ Page Splitter│    │ (Primary)    │    │ (Validator)  │           │
│  │ Region Detect│    │              │    │              │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                             │                   │                    │
│                             ▼                   ▼                    │
│                      ┌──────────────┐    ┌──────────────┐           │
│                      │ Rule Engine  │    │Cross Checker │           │
│                      │ (Secondary)  │    │ (Arbiter)    │           │
│                      └──────────────┘    └──────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Module Specifications

### 3.1 Ingestion Layer

**Responsibility:** PDF → Page Images + Raw Text

```python
# Interface: IIngestionModule
class IIngestionModule(ABC):
    @abstractmethod
    def load(self, pdf_path: Path) -> Document: ...
    
    @abstractmethod
    def render_page(self, page_num: int, dpi: int = 150) -> Image: ...
    
    @abstractmethod
    def detect_regions(self, page_num: int) -> List[Region]: ...
```

**Implementations:**
- `PdfplumberIngestion` (current)
- `PyMuPDFIngestion` (alternative, faster rendering)

### 3.2 Extraction Layer

**Responsibility:** Region → Structured Data

```python
# Interface: IExtractionEngine
class IExtractionEngine(ABC):
    @abstractmethod
    def extract_table(self, image: Image, context: Dict) -> ExtractedTable: ...
    
    @abstractmethod
    def extract_diagram(self, image: Image, context: Dict) -> ExtractedDiagram: ...
    
    @property
    @abstractmethod
    def engine_type(self) -> str: ...  # "vision" or "rule"
```

**Implementations:**
- `VisionExtractionEngine` (Primary) — GPT-4o-mini / Gemini Flash
- `RuleExtractionEngine` (Secondary) — pdfplumber + heuristics

### 3.3 Verification Layer (NEW - Double Verification)

**Responsibility:** Validate extraction results, resolve conflicts

```python
# Interface: IVerificationModule
class IVerificationModule(ABC):
    @abstractmethod
    def verify(
        self, 
        primary_result: ExtractedTable,
        secondary_result: Optional[ExtractedTable],
        raw_image: Image
    ) -> VerificationResult: ...
```

**VerificationResult:**
```python
@dataclass
class VerificationResult:
    status: str  # "verified", "conflict", "needs_review"
    confidence: float  # 0.0 - 1.0
    final_result: ExtractedTable
    conflicts: List[ConflictDetail]
    resolution_method: str  # "vision_wins", "rule_wins", "merged", "manual"
```

## 4. Double Verification Mechanism

### 4.1 Verification Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Double Verification Flow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Image ──┬──▶ Vision Engine ──▶ Result_V                        │
│          │                           │                           │
│          └──▶ Rule Engine ───▶ Result_R                         │
│                                      │                           │
│                                      ▼                           │
│                              ┌──────────────┐                    │
│                              │   Comparator │                    │
│                              └──────────────┘                    │
│                                      │                           │
│                    ┌─────────────────┼─────────────────┐        │
│                    ▼                 ▼                 ▼        │
│              [Match 100%]     [Partial Match]    [Conflict]     │
│                    │                 │                 │        │
│                    ▼                 ▼                 ▼        │
│               ✅ Verified      🔄 Merge Logic    ⚠️ Arbiter     │
│              (confidence=1.0)  (confidence=0.8)  (confidence<0.6)│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Comparison Rules

**Numeric Values:**
```python
def compare_numeric(v1: str, v2: str, tolerance: float = 0.01) -> bool:
    """
    Compare numeric values with tolerance.
    
    Examples:
        "3.3" vs "3.30" → Match
        "3.3" vs "3.31" → Match (within 1%)
        "3.3" vs "3.5" → Conflict
    """
    try:
        n1, n2 = float(v1), float(v2)
        return abs(n1 - n2) / max(abs(n1), 1e-9) <= tolerance
    except ValueError:
        return v1.strip() == v2.strip()
```

**Unit Normalization:**
```python
UNIT_ALIASES = {
    "µA": ["uA", "μA", "microamp"],
    "mA": ["milliamp"],
    "V": ["volt", "Volt"],
    "MHz": ["Mhz", "mhz"],
}

def normalize_unit(unit: str) -> str:
    """Normalize unit to canonical form."""
    for canonical, aliases in UNIT_ALIASES.items():
        if unit in aliases or unit == canonical:
            return canonical
    return unit
```

### 4.3 Conflict Resolution (Arbiter)

**Priority Rules:**
1. **Numeric precision:** Vision wins (Vision 看到的是原始图像)
2. **Unit detection:** Rule wins (规则有明确的单位映射表)
3. **Structure (rows/cols):** Vision wins (Vision 理解视觉布局)
4. **Text content:** Majority wins (如果有第三方验证)

**Escalation:**
- Confidence < 0.6 → 标记为 `needs_review`
- 关键参数冲突 (Vin, Iout, etc.) → 强制人工审核

### 4.4 Implementation Skeleton

```python
class DoubleVerifier:
    """
    Double verification implementation.
    
    Design: Modular comparators, pluggable arbiters.
    """
    
    def __init__(
        self,
        numeric_tolerance: float = 0.01,
        confidence_threshold: float = 0.60,
        critical_params: List[str] = None
    ):
        self.numeric_tolerance = numeric_tolerance
        self.confidence_threshold = confidence_threshold
        self.critical_params = critical_params or [
            "Vin", "Vout", "Iout", "Iq", "Efficiency"
        ]
        
        # Pluggable comparators
        self.comparators: Dict[str, Callable] = {
            "numeric": self._compare_numeric,
            "unit": self._compare_unit,
            "text": self._compare_text,
        }
        
        # Pluggable arbiters
        self.arbiters: Dict[str, Callable] = {
            "numeric": self._arbiter_vision_wins,
            "unit": self._arbiter_rule_wins,
            "structure": self._arbiter_vision_wins,
        }
    
    def verify(
        self,
        vision_result: ExtractedTable,
        rule_result: Optional[ExtractedTable],
        raw_image: Optional[Image] = None
    ) -> VerificationResult:
        """
        Main verification entry point.
        
        Returns:
            VerificationResult with status, confidence, and final result.
        """
        if rule_result is None:
            # No rule result available, trust Vision with lower confidence
            return VerificationResult(
                status="verified",
                confidence=0.85,
                final_result=vision_result,
                conflicts=[],
                resolution_method="vision_only"
            )
        
        # Compare results
        conflicts = self._find_conflicts(vision_result, rule_result)
        
        if not conflicts:
            return VerificationResult(
                status="verified",
                confidence=1.0,
                final_result=vision_result,
                conflicts=[],
                resolution_method="perfect_match"
            )
        
        # Resolve conflicts
        final_result, confidence = self._resolve_conflicts(
            vision_result, rule_result, conflicts
        )
        
        # Check for critical conflicts
        critical_conflicts = [
            c for c in conflicts 
            if c.field in self.critical_params and c.severity == "high"
        ]
        
        if critical_conflicts:
            return VerificationResult(
                status="needs_review",
                confidence=confidence * 0.5,
                final_result=final_result,
                conflicts=conflicts,
                resolution_method="manual_required"
            )
        
        status = "verified" if confidence >= self.confidence_threshold else "conflict"
        
        return VerificationResult(
            status=status,
            confidence=confidence,
            final_result=final_result,
            conflicts=conflicts,
            resolution_method="auto_resolved"
        )
    
    def _find_conflicts(
        self,
        v_result: ExtractedTable,
        r_result: ExtractedTable
    ) -> List[ConflictDetail]:
        """Find all conflicts between Vision and Rule results."""
        conflicts = []
        
        # Compare row by row
        for v_row, r_row in zip(v_result.rows, r_result.rows):
            for key in set(v_row.keys()) | set(r_row.keys()):
                v_val = v_row.get(key, "")
                r_val = r_row.get(key, "")
                
                if not self._values_match(v_val, r_val):
                    conflicts.append(ConflictDetail(
                        field=key,
                        vision_value=v_val,
                        rule_value=r_val,
                        severity=self._assess_severity(key, v_val, r_val)
                    ))
        
        return conflicts
    
    def _values_match(self, v1: str, v2: str) -> bool:
        """Check if two values match (with tolerance)."""
        # Try numeric comparison first
        if self._compare_numeric(v1, v2):
            return True
        
        # Fall back to normalized text comparison
        return self._normalize_text(v1) == self._normalize_text(v2)
    
    def _compare_numeric(self, v1: str, v2: str) -> bool:
        """Compare numeric values with tolerance."""
        try:
            n1 = float(re.sub(r'[^\d.\-]', '', v1))
            n2 = float(re.sub(r'[^\d.\-]', '', v2))
            return abs(n1 - n2) / max(abs(n1), 1e-9) <= self.numeric_tolerance
        except (ValueError, ZeroDivisionError):
            return False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    def _assess_severity(self, field: str, v1: str, v2: str) -> str:
        """Assess conflict severity."""
        if field in self.critical_params:
            return "high"
        
        # Check if it's a significant numeric difference
        try:
            n1, n2 = float(v1), float(v2)
            diff_pct = abs(n1 - n2) / max(abs(n1), 1e-9)
            if diff_pct > 0.1:  # >10% difference
                return "high"
            elif diff_pct > 0.05:  # >5% difference
                return "medium"
        except ValueError:
            pass
        
        return "low"
    
    def _resolve_conflicts(
        self,
        v_result: ExtractedTable,
        r_result: ExtractedTable,
        conflicts: List[ConflictDetail]
    ) -> Tuple[ExtractedTable, float]:
        """Resolve conflicts and return merged result with confidence."""
        # Start with Vision result as base
        merged = ExtractedTable(
            table_id=v_result.table_id,
            page_num=v_result.page_num,
            title=v_result.title,
            headers=v_result.headers,
            rows=[row.copy() for row in v_result.rows],
            bbox=v_result.bbox,
            confidence=v_result.confidence,
            extraction_method="double_verified"
        )
        
        resolved_count = 0
        
        for conflict in conflicts:
            arbiter = self.arbiters.get(
                self._classify_conflict(conflict),
                self._arbiter_vision_wins
            )
            
            winner = arbiter(conflict)
            
            if winner == "rule":
                # Update merged result with rule value
                for row in merged.rows:
                    if conflict.field in row:
                        row[conflict.field] = conflict.rule_value
            
            resolved_count += 1
        
        confidence = 1.0 - (len(conflicts) - resolved_count) * 0.1
        confidence = max(0.0, min(1.0, confidence))
        
        return merged, confidence
    
    def _classify_conflict(self, conflict: ConflictDetail) -> str:
        """Classify conflict type for arbiter selection."""
        if re.match(r'^[\d.\-]+$', conflict.vision_value):
            return "numeric"
        if conflict.field.lower() in ["unit", "units"]:
            return "unit"
        return "text"
    
    def _arbiter_vision_wins(self, conflict: ConflictDetail) -> str:
        return "vision"
    
    def _arbiter_rule_wins(self, conflict: ConflictDetail) -> str:
        return "rule"


@dataclass
class ConflictDetail:
    """Details of a conflict between Vision and Rule results."""
    field: str
    vision_value: str
    rule_value: str
    severity: str  # "low", "medium", "high"
```

## 5. Extensibility Points

### 5.1 Adding New Vision Models

```python
# 1. Implement IVisionClient interface
class NewModelClient(IVisionClient):
    def call(self, request: VisionRequest) -> VisionResponse:
        # Implementation
        pass

# 2. Register in factory
VisionClientFactory.register("new_model", NewModelClient)

# 3. Use via config
config = {"vision_model": "new_model"}
```

### 5.2 Adding New Verification Rules

```python
# 1. Create custom comparator
def compare_temperature(v1: str, v2: str) -> bool:
    """Compare temperature values (handle °C vs C)."""
    # Implementation
    pass

# 2. Register in verifier
verifier.comparators["temperature"] = compare_temperature
```

### 5.3 Adding New Output Formats

```python
# 1. Implement IOutputFormatter interface
class NewFormatter(IOutputFormatter):
    def format(self, extraction: DatasheetExtraction) -> str:
        # Implementation
        pass

# 2. Register in factory
OutputFormatterFactory.register("new_format", NewFormatter)
```

## 6. Configuration

```yaml
# config.yaml
extraction:
  primary_engine: "vision"
  secondary_engine: "rule"
  vision_model: "gpt-4o-mini"
  fallback_model: "gpt-4o"  # For complex tables

verification:
  enabled: true
  numeric_tolerance: 0.01
  confidence_threshold: 0.60
  critical_params:
    - "Vin"
    - "Vout"
    - "Iout"
    - "Iq"
    - "Efficiency"

output:
  format: "json"
  include_raw: false
  include_conflicts: true

cost:
  budget_per_datasheet: 0.50  # USD
  alert_threshold: 0.80  # 80% of budget
```

## 7. Testing Strategy

### 7.1 Ground Truth Dataset
- 收集 20+ 真实 datasheet
- 人工标注正确的提取结果
- 覆盖所有 edge cases (合并单元格、隐形边框、跨页等)

### 7.2 Accuracy Metrics
```python
def calculate_accuracy(extracted: Dict, ground_truth: Dict) -> float:
    """
    Calculate extraction accuracy.
    
    Metrics:
    - Cell-level accuracy: % of cells matching exactly
    - Row-level accuracy: % of rows with all cells correct
    - Table-level accuracy: % of tables with all rows correct
    """
    pass
```

### 7.3 Regression Tests
- 每次代码变更后运行完整测试集
- 精度下降 > 1% 触发告警
- CI/CD 集成

## 8. Migration Path

### Phase 1: Parallel Run (Current)
- Vision-first + Rule validation
- 记录所有冲突，不自动解决
- 收集数据，优化 arbiter 规则

### Phase 2: Auto Resolution
- 启用自动冲突解决
- 低置信度结果进入人工审核队列
- 持续优化 confidence threshold

### Phase 3: Full Production
- 99%+ 精度验证通过
- 关闭 parallel run，只保留 double verification
- 成本优化 (缓存、批处理)

---

**Document Status:** Draft v0.4
**Next Review:** After Sirius implementation feedback
