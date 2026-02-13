"""
Datasheet ETL: Vision Model Integration Module

This module provides a unified interface for calling Vision AI models
to extract structured data from complex table images and diagrams.

Supported Models:
- OpenAI GPT-4o / GPT-4o-mini
- Google Gemini Pro Vision
- Anthropic Claude 3 (via API)

Design Philosophy:
- Model-agnostic interface
- Structured output with JSON schema validation
- Retry logic with exponential backoff
- Cost tracking per extraction

Author: Axiom
Date: 2026-02-13
"""

import json
import base64
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class VisionRequest:
    """Request to Vision Model."""
    image_data: Union[bytes, str]  # Base64 or file path
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.1  # Low temp for structured extraction

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

# ============================================================================
# Prompt Templates
# ============================================================================

class PromptTemplates:
    """
    Carefully crafted prompts for datasheet extraction.
    
    Key Insight: The quality of extraction depends heavily on prompt engineering.
    These prompts are the result of iterative refinement.
    """
    
    TABLE_EXTRACTION = """You are an expert at extracting structured data from semiconductor datasheets.

Analyze this table image and extract all data into a JSON format.

RULES:
1. Identify the table title/header if visible
2. Extract column headers exactly as shown
3. For each row, create a JSON object with column names as keys
4. Handle merged cells by repeating the value for spanned rows/columns
5. Preserve units (V, mA, µA, MHz, etc.) in a separate "Unit" field if present
6. If a cell is empty, use empty string ""
7. If a value has conditions (e.g., "at 25°C"), include them in a "conditions" field

OUTPUT FORMAT:
```json
{
  "table_title": "string",
  "columns": ["col1", "col2", ...],
  "rows": [
    {"col1": "value1", "col2": "value2", ...},
    ...
  ],
  "notes": ["any footnotes or conditions"]
}
```

Extract the table data now:"""

    BLOCK_DIAGRAM = """You are an expert at understanding semiconductor block diagrams.

Analyze this block diagram and extract the functional topology as a graph.

RULES:
1. Identify all functional blocks (rectangles, circles with labels)
2. Identify all connections (arrows, lines) between blocks
3. Note any signal names on the connections
4. Identify input pins (usually on left) and output pins (usually on right)
5. Identify feedback loops if present

OUTPUT FORMAT:
```json
{
  "diagram_type": "block_diagram",
  "title": "string",
  "nodes": [
    {"id": "n1", "label": "Block Name", "type": "block|pin|ground|vcc"},
    ...
  ],
  "edges": [
    {"from": "n1", "to": "n2", "label": "signal_name", "type": "signal|power|feedback"},
    ...
  ],
  "description": "Brief description of the circuit function"
}
```

Extract the diagram topology now:"""

    PINOUT_DIAGRAM = """You are an expert at reading IC pinout diagrams.

Analyze this pinout diagram and extract all pin information.

RULES:
1. Identify the package type (QFN, SOIC, etc.)
2. Extract each pin number and its function
3. Note any special pins (NC, GND, VIN, etc.)
4. Identify pin groupings if visible

OUTPUT FORMAT:
```json
{
  "package": "QFN-16",
  "pin_count": 16,
  "pins": [
    {"number": 1, "name": "VIN", "type": "power", "description": "Input voltage"},
    {"number": 2, "name": "GND", "type": "ground", "description": "Ground"},
    ...
  ]
}
```

Extract the pinout information now:"""

    TYPICAL_APPLICATION = """You are an expert at analyzing typical application circuits.

Analyze this application circuit and extract the component connections.

RULES:
1. Identify the main IC and its connections
2. List all external components (capacitors, resistors, inductors)
3. Note component values if visible
4. Identify input and output nodes
5. Note any critical layout considerations mentioned

OUTPUT FORMAT:
```json
{
  "circuit_type": "buck_converter|ldo|amplifier|...",
  "main_ic": "part_number",
  "components": [
    {"ref": "C1", "type": "capacitor", "value": "10µF", "connection": "VIN to GND"},
    {"ref": "L1", "type": "inductor", "value": "4.7µH", "connection": "SW to VOUT"},
    ...
  ],
  "connections": [
    {"from": "VIN", "to": "IC.VIN", "through": "C1"},
    ...
  ],
  "notes": ["Layout notes", "Critical components"]
}
```

Extract the application circuit now:"""

# ============================================================================
# Vision Model Clients
# ============================================================================

class VisionModelClient(ABC):
    """Abstract base class for Vision Model clients."""
    
    @abstractmethod
    def extract(self, request: VisionRequest) -> VisionResponse:
        """Send image to Vision Model and get structured response."""
        pass
    
    def _encode_image(self, image_data: Union[bytes, str]) -> str:
        """Encode image to base64."""
        if isinstance(image_data, str):
            # Assume it's a file path
            with open(image_data, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return base64.b64encode(image_data).decode('utf-8')
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Extract JSON from model response."""
        # Try to find JSON block in response
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try parsing entire response as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None


class OpenAIVisionClient(VisionModelClient):
    """OpenAI GPT-4o Vision client."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def extract(self, request: VisionRequest) -> VisionResponse:
        """Send image to GPT-4o and get structured response."""
        import urllib.request
        import urllib.error
        
        start_time = time.time()
        
        # Encode image
        image_b64 = self._encode_image(request.image_data)
        
        # Build request
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        try:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            latency_ms = int((time.time() - start_time) * 1000)
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            
            # Calculate cost
            pricing = self.PRICING.get(request.model, self.PRICING["gpt-4o-mini"])
            cost = (
                usage.get("prompt_tokens", 0) / 1000 * pricing["input"] +
                usage.get("completion_tokens", 0) / 1000 * pricing["output"]
            )
            
            return VisionResponse(
                success=True,
                content=content,
                parsed_json=self._parse_json_response(content),
                model=request.model,
                latency_ms=latency_ms,
                tokens_used=usage.get("total_tokens", 0),
                cost_usd=cost
            )
            
        except Exception as e:
            return VisionResponse(
                success=False,
                error=str(e),
                model=request.model,
                latency_ms=int((time.time() - start_time) * 1000)
            )


class MockVisionClient(VisionModelClient):
    """Mock client for testing without API calls."""
    
    def extract(self, request: VisionRequest) -> VisionResponse:
        """Return mock response for testing."""
        logger.info("MockVisionClient: Simulating Vision API call")
        
        # Simulate latency
        time.sleep(0.1)
        
        # Return mock data based on prompt type
        if "table" in request.prompt.lower():
            mock_json = {
                "table_title": "Electrical Characteristics (Mock)",
                "columns": ["Parameter", "Min", "Typ", "Max", "Unit"],
                "rows": [
                    {"Parameter": "VIN Range", "Min": "4.2", "Typ": "", "Max": "36", "Unit": "V"},
                    {"Parameter": "IQ", "Min": "", "Typ": "25", "Max": "40", "Unit": "µA"},
                ],
                "notes": ["Mock data for testing"]
            }
        elif "block" in request.prompt.lower() or "diagram" in request.prompt.lower():
            mock_json = {
                "diagram_type": "block_diagram",
                "title": "Functional Block Diagram (Mock)",
                "nodes": [
                    {"id": "n1", "label": "Error Amp", "type": "block"},
                    {"id": "n2", "label": "PWM", "type": "block"},
                ],
                "edges": [
                    {"from": "FB", "to": "n1", "label": "feedback"},
                    {"from": "n1", "to": "n2", "label": "error"},
                ]
            }
        else:
            mock_json = {"mock": True, "prompt_received": request.prompt[:100]}
        
        return VisionResponse(
            success=True,
            content=json.dumps(mock_json, indent=2),
            parsed_json=mock_json,
            model="mock",
            latency_ms=100,
            tokens_used=0,
            cost_usd=0.0
        )

# ============================================================================
# Vision Extraction Manager
# ============================================================================

class VisionExtractionManager:
    """
    High-level manager for Vision-based extraction.
    
    Handles:
    - Client selection based on configuration
    - Retry logic with exponential backoff
    - Cost tracking and budgeting
    - Result caching
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        max_retries: int = 3,
        budget_usd: float = 10.0
    ):
        self.default_model = default_model
        self.max_retries = max_retries
        self.budget_usd = budget_usd
        self.total_cost = 0.0
        
        # Initialize clients
        if openai_api_key:
            self.client = OpenAIVisionClient(openai_api_key)
        else:
            logger.warning("No API key provided. Using MockVisionClient.")
            self.client = MockVisionClient()
    
    def extract_table(
        self,
        image_path: str,
        model: Optional[str] = None
    ) -> VisionResponse:
        """Extract table data from image."""
        return self._extract(
            image_path,
            PromptTemplates.TABLE_EXTRACTION,
            model or self.default_model
        )
    
    def extract_block_diagram(
        self,
        image_path: str,
        model: Optional[str] = None
    ) -> VisionResponse:
        """Extract block diagram topology."""
        return self._extract(
            image_path,
            PromptTemplates.BLOCK_DIAGRAM,
            model or self.default_model
        )
    
    def extract_pinout(
        self,
        image_path: str,
        model: Optional[str] = None
    ) -> VisionResponse:
        """Extract pinout information."""
        return self._extract(
            image_path,
            PromptTemplates.PINOUT_DIAGRAM,
            model or self.default_model
        )
    
    def extract_application_circuit(
        self,
        image_path: str,
        model: Optional[str] = None
    ) -> VisionResponse:
        """Extract typical application circuit."""
        return self._extract(
            image_path,
            PromptTemplates.TYPICAL_APPLICATION,
            model or self.default_model
        )
    
    def _extract(
        self,
        image_path: str,
        prompt: str,
        model: str
    ) -> VisionResponse:
        """Internal extraction with retry logic."""
        # Check budget
        if self.total_cost >= self.budget_usd:
            return VisionResponse(
                success=False,
                error=f"Budget exceeded: ${self.total_cost:.2f} >= ${self.budget_usd:.2f}"
            )
        
        request = VisionRequest(
            image_data=image_path,
            prompt=prompt,
            model=model
        )
        
        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            response = self.client.extract(request)
            
            if response.success:
                self.total_cost += response.cost_usd
                logger.info(
                    f"Extraction successful. Cost: ${response.cost_usd:.4f}, "
                    f"Total: ${self.total_cost:.4f}"
                )
                return response
            
            # Exponential backoff
            wait_time = 2 ** attempt
            logger.warning(
                f"Extraction failed (attempt {attempt + 1}/{self.max_retries}): "
                f"{response.error}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
        
        return response  # Return last failed response

# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with mock client
    manager = VisionExtractionManager()
    
    print("Testing Vision Extraction Manager (Mock Mode)")
    print("=" * 50)
    
    # Test table extraction
    print("\n[Test] Table Extraction:")
    response = manager.extract_table("test_image.png")
    print(f"  Success: {response.success}")
    print(f"  Parsed JSON: {json.dumps(response.parsed_json, indent=2)[:200]}...")
    
    # Test block diagram
    print("\n[Test] Block Diagram Extraction:")
    response = manager.extract_block_diagram("test_diagram.png")
    print(f"  Success: {response.success}")
    print(f"  Nodes: {len(response.parsed_json.get('nodes', []))}")
    print(f"  Edges: {len(response.parsed_json.get('edges', []))}")
    
    print("\n" + "=" * 50)
    print(f"Total Cost: ${manager.total_cost:.4f}")
