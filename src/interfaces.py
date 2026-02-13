"""
Datasheet ETL: Core Interfaces (v0.4)

This module defines the abstract interfaces for the modular ETL pipeline.
All implementations must conform to these interfaces for interoperability.

Design Goals:
- Dependency Inversion: High-level modules depend on abstractions
- Open/Closed: Open for extension, closed for modification
- Single Responsibility: Each interface has one clear purpose

Author: Axiom
Date: 2026-02-13
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from enum import Enum

# Try to import PIL, but don't fail if not available
try:
    from PIL import Image
except ImportError:
    Image = Any  # Type hint fallback


# ============================================================================
# Enums
# ============================================================================

class ExtractionMethod(Enum):
    """Extraction method identifier."""
    VISION = "vision"
    RULE = "rule"
    HYBRID = "hybrid"
    DOUBLE_VERIFIED = "double_verified"


class VerificationStatus(Enum):
    """Verification result status."""
    VERIFIED = "verified"
    CONFLICT = "conflict"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ConflictSeverity(Enum):
    """Conflict severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Region:
    """A detected region in a PDF page."""
    region_id: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    region_type: str  # "table", "diagram", "text", "unknown"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTable:
    """Represents an extracted table with metadata."""
    table_id: str
    page_num: int
    title: str
    headers: List[str]
    rows: List[Dict[str, str]]
    bbox: Tuple[float, float, float, float]
    confidence: float
    extraction_method: ExtractionMethod
    raw_data: Optional[List[List[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictDetail:
    """Details of a conflict between two extraction results."""
    field: str
    primary_value: str
    secondary_value: str
    severity: ConflictSeverity
    resolution: Optional[str] = None  # "primary_wins", "secondary_wins", "merged"
    resolved_value: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of double verification."""
    status: VerificationStatus
    confidence: float  # 0.0 - 1.0
    final_result: Union[ExtractedTable, ExtractedDiagram]
    conflicts: List[ConflictDetail] = field(default_factory=list)
    resolution_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Represents a loaded PDF document."""
    source_path: Path
    total_pages: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Internal handle (implementation-specific)
    _handle: Any = None


@dataclass
class DatasheetExtraction:
    """Complete extraction result for a datasheet."""
    source_file: str
    mpn: str
    manufacturer: str
    total_pages: int
    tables: List[ExtractedTable] = field(default_factory=list)
    diagrams: List[ExtractedDiagram] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    verification_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Ingestion Layer Interfaces
# ============================================================================

class IIngestionModule(ABC):
    """
    Interface for PDF ingestion modules.
    
    Responsibility: Load PDF, render pages, detect regions.
    """
    
    @abstractmethod
    def load(self, pdf_path: Path) -> Document:
        """
        Load a PDF document.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Document object with metadata.
        """
        pass
    
    @abstractmethod
    def render_page(
        self, 
        doc: Document, 
        page_num: int, 
        dpi: int = 150
    ) -> "Image":
        """
        Render a page as an image.
        
        Args:
            doc: Loaded document.
            page_num: Page number (0-indexed).
            dpi: Resolution for rendering.
            
        Returns:
            PIL Image object.
        """
        pass
    
    @abstractmethod
    def detect_regions(
        self, 
        doc: Document, 
        page_num: int
    ) -> List[Region]:
        """
        Detect regions (tables, diagrams) in a page.
        
        Args:
            doc: Loaded document.
            page_num: Page number (0-indexed).
            
        Returns:
            List of detected regions.
        """
        pass
    
    @abstractmethod
    def extract_text(
        self, 
        doc: Document, 
        page_num: int
    ) -> str:
        """
        Extract raw text from a page.
        
        Args:
            doc: Loaded document.
            page_num: Page number (0-indexed).
            
        Returns:
            Extracted text.
        """
        pass
    
    def close(self, doc: Document) -> None:
        """
        Close the document and release resources.
        
        Default implementation does nothing.
        """
        pass


# ============================================================================
# Extraction Layer Interfaces
# ============================================================================

class IExtractionEngine(ABC):
    """
    Interface for extraction engines.
    
    Responsibility: Extract structured data from images/regions.
    """
    
    @property
    @abstractmethod
    def engine_type(self) -> ExtractionMethod:
        """Return the engine type (vision/rule)."""
        pass
    
    @abstractmethod
    def extract_table(
        self, 
        image: "Image",
        context: Optional[Dict[str, Any]] = None
    ) -> ExtractedTable:
        """
        Extract table data from an image.
        
        Args:
            image: PIL Image of the table region.
            context: Optional context (page text, surrounding content).
            
        Returns:
            ExtractedTable with structured data.
        """
        pass
    
    @abstractmethod
    def extract_diagram(
        self, 
        image: "Image",
        context: Optional[Dict[str, Any]] = None
    ) -> ExtractedDiagram:
        """
        Extract diagram topology from an image.
        
        Args:
            image: PIL Image of the diagram region.
            context: Optional context.
            
        Returns:
            ExtractedDiagram with nodes and edges.
        """
        pass
    
    def supports_batch(self) -> bool:
        """Return True if engine supports batch processing."""
        return False
    
    def extract_batch(
        self, 
        items: List[Tuple["Image", Dict]]
    ) -> List[Union[ExtractedTable, ExtractedDiagram]]:
        """
        Batch extraction (optional).
        
        Default implementation calls extract_table/diagram sequentially.
        """
        raise NotImplementedError("Batch extraction not supported")


# ============================================================================
# Verification Layer Interfaces
# ============================================================================

class IVerificationModule(ABC):
    """
    Interface for verification modules.
    
    Responsibility: Validate and reconcile extraction results.
    """
    
    @abstractmethod
    def verify_table(
        self,
        primary_result: ExtractedTable,
        secondary_result: Optional[ExtractedTable] = None,
        raw_image: Optional["Image"] = None
    ) -> VerificationResult:
        """
        Verify table extraction result.
        
        Args:
            primary_result: Result from primary engine (Vision).
            secondary_result: Result from secondary engine (Rule).
            raw_image: Original image for re-verification if needed.
            
        Returns:
            VerificationResult with status and final result.
        """
        pass
    
    @abstractmethod
    def verify_diagram(
        self,
        primary_result: ExtractedDiagram,
        secondary_result: Optional[ExtractedDiagram] = None,
        raw_image: Optional["Image"] = None
    ) -> VerificationResult:
        """
        Verify diagram extraction result.
        
        Args:
            primary_result: Result from primary engine.
            secondary_result: Result from secondary engine.
            raw_image: Original image for re-verification if needed.
            
        Returns:
            VerificationResult with status and final result.
        """
        pass


class IComparator(ABC):
    """
    Interface for value comparators.
    
    Responsibility: Compare two values and determine if they match.
    """
    
    @abstractmethod
    def compare(self, value1: str, value2: str) -> bool:
        """
        Compare two values.
        
        Args:
            value1: First value.
            value2: Second value.
            
        Returns:
            True if values match (within tolerance).
        """
        pass
    
    @property
    @abstractmethod
    def comparator_type(self) -> str:
        """Return comparator type identifier."""
        pass


class IArbiter(ABC):
    """
    Interface for conflict arbiters.
    
    Responsibility: Decide which value wins in a conflict.
    """
    
    @abstractmethod
    def arbitrate(self, conflict: ConflictDetail) -> str:
        """
        Decide which value wins.
        
        Args:
            conflict: Conflict details.
            
        Returns:
            "primary" or "secondary"
        """
        pass
    
    @property
    @abstractmethod
    def arbiter_type(self) -> str:
        """Return arbiter type identifier."""
        pass


# ============================================================================
# Output Layer Interfaces
# ============================================================================

class IOutputFormatter(ABC):
    """
    Interface for output formatters.
    
    Responsibility: Format extraction results for output.
    """
    
    @abstractmethod
    def format(self, extraction: DatasheetExtraction) -> str:
        """
        Format extraction result.
        
        Args:
            extraction: Complete extraction result.
            
        Returns:
            Formatted string (JSON, CSV, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def format_type(self) -> str:
        """Return format type identifier."""
        pass


# ============================================================================
# Vision Client Interface
# ============================================================================

@dataclass
class VisionRequest:
    """Request to Vision Model."""
    image_data: Union[bytes, str, Path]  # Base64, file path, or raw bytes
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionResponse:
    """Response from Vision Model."""
    success: bool
    content: Optional[str] = None
    parsed_json: Optional[Dict] = None
    error: Optional[str] = None
    model: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0


class IVisionClient(ABC):
    """
    Interface for Vision Model clients.
    
    Responsibility: Call Vision APIs and return structured responses.
    """
    
    @abstractmethod
    def call(self, request: VisionRequest) -> VisionResponse:
        """
        Call Vision Model.
        
        Args:
            request: Vision request with image and prompt.
            
        Returns:
            VisionResponse with extracted content.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass
    
    def supports_batch(self) -> bool:
        """Return True if client supports batch requests."""
        return False


# ============================================================================
# Factory Interfaces
# ============================================================================

class IFactory(ABC):
    """Generic factory interface."""
    
    @classmethod
    @abstractmethod
    def create(cls, type_name: str, **kwargs) -> Any:
        """Create an instance by type name."""
        pass
    
    @classmethod
    @abstractmethod
    def register(cls, type_name: str, implementation: type) -> None:
        """Register a new implementation."""
        pass
    
    @classmethod
    @abstractmethod
    def list_types(cls) -> List[str]:
        """List all registered types."""
        pass


# ============================================================================
# Pipeline Interface
# ============================================================================

class IPipeline(ABC):
    """
    Interface for the complete ETL pipeline.
    
    Responsibility: Orchestrate ingestion, extraction, verification, output.
    """
    
    @abstractmethod
    def process(
        self, 
        pdf_path: Path,
        output_path: Optional[Path] = None
    ) -> DatasheetExtraction:
        """
        Process a datasheet PDF.
        
        Args:
            pdf_path: Path to input PDF.
            output_path: Optional path for output file.
            
        Returns:
            Complete extraction result.
        """
        pass
    
    @abstractmethod
    def process_batch(
        self,
        pdf_paths: List[Path],
        output_dir: Optional[Path] = None
    ) -> List[DatasheetExtraction]:
        """
        Process multiple datasheets.
        
        Args:
            pdf_paths: List of input PDF paths.
            output_dir: Optional directory for output files.
            
        Returns:
            List of extraction results.
        """
        pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the ETL pipeline."""
    
    # Extraction
    primary_engine: str = "vision"
    secondary_engine: str = "rule"
    vision_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o"
    
    # Verification
    verification_enabled: bool = True
    numeric_tolerance: float = 0.01
    confidence_threshold: float = 0.60
    critical_params: List[str] = field(default_factory=lambda: [
        "Vin", "Vout", "Iout", "Iq", "Efficiency"
    ])
    
    # Output
    output_format: str = "json"
    include_raw: bool = False
    include_conflicts: bool = True
    
    # Cost control
    budget_per_datasheet: float = 0.50
    alert_threshold: float = 0.80
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
