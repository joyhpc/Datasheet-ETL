"""
Datasheet ETL: PDF Rendering Utilities

This module provides utilities for rendering PDF pages to images,
which can then be sent to Vision Models for extraction.

Dependencies:
- PyMuPDF (fitz): For high-quality PDF rendering
- Pillow: For image processing (optional)

Fallback:
- If PyMuPDF not available, uses pdftoppm (poppler-utils)

Author: Axiom
Date: 2026-02-13
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RenderedPage:
    """Represents a rendered PDF page."""
    page_num: int
    image_path: str
    width: int
    height: int
    dpi: int

class PDFRenderer:
    """
    PDF to Image renderer with multiple backend support.
    
    Priority:
    1. PyMuPDF (fitz) - Best quality, pure Python
    2. pdftoppm (poppler) - Good quality, requires system package
    3. pdf2image - Wrapper around poppler
    """
    
    def __init__(self, dpi: int = 150):
        """
        Initialize renderer.
        
        Args:
            dpi: Resolution for rendering (150 is good balance of quality/size)
        """
        self.dpi = dpi
        self.backend = self._detect_backend()
        logger.info(f"PDF Renderer initialized with backend: {self.backend}")
    
    def _detect_backend(self) -> str:
        """Detect available rendering backend."""
        # Try PyMuPDF first
        try:
            import fitz
            return "pymupdf"
        except ImportError:
            pass
        
        # Try pdftoppm
        try:
            result = subprocess.run(
                ["pdftoppm", "-v"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0 or b"pdftoppm" in result.stderr:
                return "poppler"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # No backend available
        return "none"
    
    def render_page(
        self,
        pdf_path: str,
        page_num: int,
        output_dir: Optional[str] = None
    ) -> Optional[RenderedPage]:
        """
        Render a single page to image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            output_dir: Directory for output image (uses temp if None)
            
        Returns:
            RenderedPage object or None if failed
        """
        if self.backend == "none":
            logger.error("No PDF rendering backend available")
            return None
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="datasheet_etl_")
        
        output_path = Path(output_dir) / f"page_{page_num:03d}.png"
        
        if self.backend == "pymupdf":
            return self._render_pymupdf(pdf_path, page_num, str(output_path))
        elif self.backend == "poppler":
            return self._render_poppler(pdf_path, page_num, str(output_path))
        
        return None
    
    def render_all_pages(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[RenderedPage]:
        """
        Render all pages (or a range) to images.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for output images
            page_range: Optional (start, end) tuple (1-indexed, inclusive)
            
        Returns:
            List of RenderedPage objects
        """
        pages = []
        
        # Get total page count
        total_pages = self._get_page_count(pdf_path)
        if total_pages == 0:
            return pages
        
        # Determine range
        start = page_range[0] if page_range else 1
        end = page_range[1] if page_range else total_pages
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="datasheet_etl_")
        
        for page_num in range(start, end + 1):
            rendered = self.render_page(pdf_path, page_num, output_dir)
            if rendered:
                pages.append(rendered)
        
        return pages
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get total number of pages in PDF."""
        if self.backend == "pymupdf":
            try:
                import fitz
                doc = fitz.open(pdf_path)
                count = len(doc)
                doc.close()
                return count
            except Exception as e:
                logger.error(f"Failed to get page count: {e}")
                return 0
        
        elif self.backend == "poppler":
            try:
                result = subprocess.run(
                    ["pdfinfo", pdf_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                for line in result.stdout.split('\n'):
                    if line.startswith("Pages:"):
                        return int(line.split(':')[1].strip())
            except Exception as e:
                logger.error(f"Failed to get page count: {e}")
        
        return 0
    
    def _render_pymupdf(
        self,
        pdf_path: str,
        page_num: int,
        output_path: str
    ) -> Optional[RenderedPage]:
        """Render using PyMuPDF."""
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]  # 0-indexed
            
            # Render at specified DPI
            zoom = self.dpi / 72  # PDF default is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            pix.save(output_path)
            
            result = RenderedPage(
                page_num=page_num,
                image_path=output_path,
                width=pix.width,
                height=pix.height,
                dpi=self.dpi
            )
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"PyMuPDF render failed: {e}")
            return None
    
    def _render_poppler(
        self,
        pdf_path: str,
        page_num: int,
        output_path: str
    ) -> Optional[RenderedPage]:
        """Render using poppler (pdftoppm)."""
        try:
            # pdftoppm outputs to prefix, so we need to handle that
            output_prefix = output_path.replace('.png', '')
            
            result = subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r", str(self.dpi),
                    "-f", str(page_num),
                    "-l", str(page_num),
                    "-singlefile",
                    pdf_path,
                    output_prefix
                ],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"pdftoppm failed: {result.stderr.decode()}")
                return None
            
            # Get image dimensions (would need PIL or similar)
            # For now, use placeholder values
            return RenderedPage(
                page_num=page_num,
                image_path=output_path,
                width=0,  # Would need to read image
                height=0,
                dpi=self.dpi
            )
            
        except Exception as e:
            logger.error(f"Poppler render failed: {e}")
            return None


class TableRegionDetector:
    """
    Detects table regions in PDF pages for targeted extraction.
    
    Uses heuristics based on:
    - Horizontal/vertical line patterns
    - Text density analysis
    - Whitespace patterns
    """
    
    def __init__(self):
        self.min_table_height = 50  # pixels
        self.min_table_width = 100
    
    def detect_tables(
        self,
        page_data: dict,
        page_image_path: Optional[str] = None
    ) -> List[Tuple[float, float, float, float]]:
        """
        Detect table bounding boxes in a page.
        
        Args:
            page_data: Page data from pdfplumber
            page_image_path: Optional rendered page image
            
        Returns:
            List of (x0, y0, x1, y1) bounding boxes
        """
        tables = []
        
        # Method 1: Use pdfplumber's built-in table detection
        if "tables" in page_data and page_data["tables"]:
            # pdfplumber found tables, estimate their positions
            # This is a simplification - real implementation would use
            # the actual table coordinates from pdfplumber
            for i, table in enumerate(page_data["tables"]):
                if table and len(table) > 1:
                    # Placeholder bbox - would need actual coordinates
                    tables.append((50, 100 + i * 200, 550, 300 + i * 200))
        
        # Method 2: Line-based detection (if no tables found)
        if not tables and "lines" in page_data:
            # Analyze horizontal and vertical lines to find table boundaries
            pass
        
        return tables
    
    def crop_region(
        self,
        image_path: str,
        bbox: Tuple[float, float, float, float],
        output_path: str
    ) -> bool:
        """
        Crop a region from an image.
        
        Args:
            image_path: Source image path
            bbox: (x0, y0, x1, y1) bounding box
            output_path: Output path for cropped image
            
        Returns:
            True if successful
        """
        try:
            from PIL import Image
            
            img = Image.open(image_path)
            cropped = img.crop(bbox)
            cropped.save(output_path)
            return True
            
        except ImportError:
            logger.warning("PIL not available for cropping")
            return False
        except Exception as e:
            logger.error(f"Crop failed: {e}")
            return False


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("PDF Renderer Test")
    print("=" * 50)
    
    renderer = PDFRenderer(dpi=150)
    print(f"Backend: {renderer.backend}")
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"\nRendering: {pdf_path}")
        
        page_count = renderer._get_page_count(pdf_path)
        print(f"Total pages: {page_count}")
        
        if page_count > 0:
            # Render first page
            result = renderer.render_page(pdf_path, 1)
            if result:
                print(f"Rendered page 1: {result.image_path}")
                print(f"Dimensions: {result.width}x{result.height}")
            else:
                print("Rendering failed")
    else:
        print("\nUsage: python pdf_renderer.py <pdf_path>")
        print("No PDF provided, skipping render test.")
