"""
Datasheet ETL: Double Verification Module

This module implements the Double Verification mechanism for ensuring
99%+ extraction accuracy by cross-checking Vision and Rule-based results.

Design Philosophy:
- Vision-first: Vision Model is the primary extractor
- Rule-based validation: Rules catch Vision hallucinations
- Pluggable arbiters: Different conflict types use different resolution strategies
- Confidence scoring: Low confidence results go to manual review queue

Author: Axiom
Date: 2026-02-13
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Models
# ============================================================================

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


@dataclass
class ConflictDetail:
    """Details of a conflict between Vision and Rule results."""
    field: str
    vision_value: str
    rule_value: str
    severity: ConflictSeverity
    conflict_type: str = "unknown"  # "numeric", "unit", "text", "structure"
    resolution: Optional[str] = None  # "vision_wins", "rule_wins", "merged"
    resolved_value: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of double verification."""
    status: VerificationStatus
    confidence: float  # 0.0 - 1.0
    final_result: Dict[str, Any]
    conflicts: List[ConflictDetail] = field(default_factory=list)
    resolution_method: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Comparators (Pluggable)
# ============================================================================

class NumericComparator:
    """Compare numeric values with tolerance."""
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def compare(self, v1: str, v2: str) -> bool:
        """
        Compare two values that may be numeric.
        
        Returns True if values match within tolerance.
        """
        # Try to extract numeric values
        n1 = self._extract_number(v1)
        n2 = self._extract_number(v2)
        
        if n1 is None or n2 is None:
            return False
        
        # Handle zero case
        if n1 == 0 and n2 == 0:
            return True
        
        # Calculate relative difference
        max_val = max(abs(n1), abs(n2))
        if max_val == 0:
            return True
        
        diff = abs(n1 - n2) / max_val
        return diff <= self.tolerance
    
    def _extract_number(self, value: str) -> Optional[float]:
        """Extract numeric value from string."""
        if not value:
            return None
        
        # Remove common non-numeric characters but keep decimal and sign
        cleaned = re.sub(r'[^\d.\-+eE]', '', str(value).strip())
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None


class UnitComparator:
    """Compare unit strings with normalization."""
    
    UNIT_ALIASES = {
        "µA": ["uA", "μA", "microamp", "micro-amp", "ua"],
        "mA": ["milliamp", "milli-amp", "ma"],
        "A": ["amp", "amps", "ampere"],
        "µV": ["uV", "μV", "microvolt"],
        "mV": ["millivolt", "mv"],
        "V": ["volt", "volts", "Volt"],
        "kV": ["kilovolt", "kv"],
        "µW": ["uW", "μW", "microwatt"],
        "mW": ["milliwatt", "mw"],
        "W": ["watt", "watts"],
        "kW": ["kilowatt", "kw"],
        "µF": ["uF", "μF", "microfarad"],
        "nF": ["nanofarad", "nf"],
        "pF": ["picofarad", "pf"],
        "µH": ["uH", "μH", "microhenry"],
        "mH": ["millihenry", "mh"],
        "nH": ["nanohenry", "nh"],
        "Ω": ["ohm", "ohms", "R"],
        "kΩ": ["kohm", "k-ohm", "kR"],
        "MΩ": ["Mohm", "megohm", "MR"],
        "Hz": ["hertz", "hz"],
        "kHz": ["kilohertz", "khz"],
        "MHz": ["megahertz", "mhz", "Mhz"],
        "GHz": ["gigahertz", "ghz"],
        "°C": ["C", "degC", "deg C", "celsius"],
        "°F": ["F", "degF", "deg F", "fahrenheit"],
        "ns": ["nanosecond", "nanosec"],
        "µs": ["us", "μs", "microsecond", "microsec"],
        "ms": ["millisecond", "millisec"],
        "s": ["sec", "second", "seconds"],
    }
    
    def __init__(self):
        # Build reverse lookup
        self._reverse_map = {}
        for canonical, aliases in self.UNIT_ALIASES.items():
            self._reverse_map[canonical.lower()] = canonical
            for alias in aliases:
                self._reverse_map[alias.lower()] = canonical
    
    def compare(self, v1: str, v2: str) -> bool:
        """Compare two unit strings."""
        n1 = self.normalize(v1)
        n2 = self.normalize(v2)
        return n1 == n2
    
    def normalize(self, unit: str) -> str:
        """Normalize unit to canonical form."""
        if not unit:
            return ""
        
        cleaned = unit.strip()
        lookup = cleaned.lower()
        
        return self._reverse_map.get(lookup, cleaned)


class TextComparator:
    """Compare text strings with normalization."""
    
    def compare(self, v1: str, v2: str) -> bool:
        """Compare two text strings after normalization."""
        n1 = self._normalize(v1)
        n2 = self._normalize(v2)
        return n1 == n2
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Lowercase, collapse whitespace, strip
        normalized = re.sub(r'\s+', ' ', str(text).strip().lower())
        
        # Remove common punctuation variations
        normalized = re.sub(r'[.,;:!?]', '', normalized)
        
        return normalized


# ============================================================================
# Arbiters (Pluggable)
# ============================================================================

class VisionWinsArbiter:
    """Arbiter that always chooses Vision result."""
    
    def arbitrate(self, conflict: ConflictDetail) -> str:
        return "vision_wins"


class RuleWinsArbiter:
    """Arbiter that always chooses Rule result."""
    
    def arbitrate(self, conflict: ConflictDetail) -> str:
        return "rule_wins"


class SmartArbiter:
    """
    Smart arbiter that chooses based on conflict type.
    
    Strategy:
    - Numeric precision: Vision wins (sees original image)
    - Unit detection: Rule wins (has explicit unit mapping)
    - Structure (rows/cols): Vision wins (understands visual layout)
    - Text content: Vision wins (can read complex fonts)
    """
    
    def arbitrate(self, conflict: ConflictDetail) -> str:
        if conflict.conflict_type == "unit":
            return "rule_wins"
        else:
            # numeric, text, structure, unknown
            return "vision_wins"


# ============================================================================
# Double Verifier
# ============================================================================

class DoubleVerifier:
    """
    Double Verification implementation.
    
    Compares Vision and Rule extraction results, identifies conflicts,
    and resolves them using pluggable arbiters.
    
    Usage:
        verifier = DoubleVerifier()
        result = verifier.verify(vision_table, rule_table)
        
        if result.status == VerificationStatus.NEEDS_REVIEW:
            # Send to manual review queue
            pass
    """
    
    # Critical parameters that require extra scrutiny
    DEFAULT_CRITICAL_PARAMS = [
        "vin", "vout", "iout", "iq", "efficiency",
        "vin_min", "vin_max", "vout_min", "vout_max",
        "iout_max", "power", "frequency"
    ]
    
    def __init__(
        self,
        numeric_tolerance: float = 0.01,
        confidence_threshold: float = 0.60,
        critical_params: Optional[List[str]] = None
    ):
        """
        Initialize the Double Verifier.
        
        Args:
            numeric_tolerance: Tolerance for numeric comparisons (default 1%)
            confidence_threshold: Below this, mark as needs_review
            critical_params: List of critical parameter names
        """
        self.numeric_tolerance = numeric_tolerance
        self.confidence_threshold = confidence_threshold
        self.critical_params = [
            p.lower() for p in (critical_params or self.DEFAULT_CRITICAL_PARAMS)
        ]
        
        # Initialize comparators
        self.numeric_comparator = NumericComparator(tolerance=numeric_tolerance)
        self.unit_comparator = UnitComparator()
        self.text_comparator = TextComparator()
        
        # Initialize arbiter
        self.arbiter = SmartArbiter()
        
        # Stats tracking
        self._stats = {
            "total_verifications": 0,
            "verified_count": 0,
            "conflict_count": 0,
            "needs_review_count": 0,
            "total_conflicts_found": 0,
            "conflicts_resolved": 0,
        }
    
    def verify(
        self,
        vision_result: Dict[str, Any],
        rule_result: Optional[Dict[str, Any]] = None,
        raw_image: Optional[Any] = None
    ) -> VerificationResult:
        """
        Verify extraction results using double verification.
        
        Args:
            vision_result: Result from Vision extraction (primary)
            rule_result: Result from Rule-based extraction (secondary)
            raw_image: Original image for re-verification if needed
            
        Returns:
            VerificationResult with status, confidence, and final result
        """
        self._stats["total_verifications"] += 1
        
        # If no rule result, trust Vision with reduced confidence
        if rule_result is None:
            self._stats["verified_count"] += 1
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.85,
                final_result=vision_result,
                conflicts=[],
                resolution_method="vision_only",
                stats={"vision_only": True}
            )
        
        # Find conflicts
        conflicts = self._find_conflicts(vision_result, rule_result)
        self._stats["total_conflicts_found"] += len(conflicts)
        
        # Perfect match
        if not conflicts:
            self._stats["verified_count"] += 1
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=1.0,
                final_result=vision_result,
                conflicts=[],
                resolution_method="perfect_match",
                stats={"perfect_match": True}
            )
        
        # Resolve conflicts
        final_result, resolved_conflicts = self._resolve_conflicts(
            vision_result, rule_result, conflicts
        )
        self._stats["conflicts_resolved"] += len(resolved_conflicts)
        
        # Calculate confidence
        confidence = self._calculate_confidence(conflicts, resolved_conflicts)
        
        # Check for critical conflicts (even if resolved, large differences need review)
        critical_conflicts = [
            c for c in conflicts
            if c.severity == ConflictSeverity.CRITICAL
        ]
        
        # Check for unresolved high-severity conflicts
        unresolved_high = [
            c for c in conflicts
            if c.severity in (ConflictSeverity.HIGH, ConflictSeverity.CRITICAL)
            and c.resolution is None
        ]
        
        # Determine status
        if unresolved_high or (critical_conflicts and self._has_large_difference(critical_conflicts)):
            self._stats["needs_review_count"] += 1
            status = VerificationStatus.NEEDS_REVIEW
            resolution_method = "manual_required"
        elif confidence < self.confidence_threshold:
            self._stats["conflict_count"] += 1
            status = VerificationStatus.CONFLICT
            resolution_method = "low_confidence"
        else:
            self._stats["verified_count"] += 1
            status = VerificationStatus.VERIFIED
            resolution_method = "auto_resolved"
        
        return VerificationResult(
            status=status,
            confidence=confidence,
            final_result=final_result,
            conflicts=resolved_conflicts,
            resolution_method=resolution_method,
            stats={
                "total_conflicts": len(conflicts),
                "critical_conflicts": len(critical_conflicts),
                "resolved_conflicts": len([c for c in resolved_conflicts if c.resolution])
            }
        )
    
    def verify_table(
        self,
        vision_table: Dict[str, Any],
        rule_table: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify table extraction results.
        
        Specialized method for table verification with row-by-row comparison.
        """
        return self.verify(vision_table, rule_table)
    
    def _find_conflicts(
        self,
        vision_result: Dict[str, Any],
        rule_result: Dict[str, Any]
    ) -> List[ConflictDetail]:
        """Find all conflicts between Vision and Rule results."""
        conflicts = []
        
        # Compare rows if present
        v_rows = vision_result.get("rows", [])
        r_rows = rule_result.get("rows", [])
        
        # Compare row by row
        for i, (v_row, r_row) in enumerate(zip(v_rows, r_rows)):
            if not isinstance(v_row, dict) or not isinstance(r_row, dict):
                continue
            
            all_keys = set(v_row.keys()) | set(r_row.keys())
            
            for key in all_keys:
                v_val = str(v_row.get(key, ""))
                r_val = str(r_row.get(key, ""))
                
                if not self._values_match(v_val, r_val):
                    conflict_type = self._classify_conflict(key, v_val, r_val)
                    # Pass row context for critical parameter detection
                    severity = self._assess_severity(key, v_val, r_val, row_context=v_row)
                    
                    conflicts.append(ConflictDetail(
                        field=f"row[{i}].{key}",
                        vision_value=v_val,
                        rule_value=r_val,
                        severity=severity,
                        conflict_type=conflict_type
                    ))
        
        # Compare headers
        v_headers = vision_result.get("headers", [])
        r_headers = rule_result.get("headers", [])
        
        if v_headers != r_headers:
            conflicts.append(ConflictDetail(
                field="headers",
                vision_value=str(v_headers),
                rule_value=str(r_headers),
                severity=ConflictSeverity.MEDIUM,
                conflict_type="structure"
            ))
        
        # Compare metadata
        for key in ["title", "table_id"]:
            v_val = str(vision_result.get(key, ""))
            r_val = str(rule_result.get(key, ""))
            
            if v_val and r_val and not self._values_match(v_val, r_val):
                conflicts.append(ConflictDetail(
                    field=key,
                    vision_value=v_val,
                    rule_value=r_val,
                    severity=ConflictSeverity.LOW,
                    conflict_type="text"
                ))
        
        return conflicts
    
    def _values_match(self, v1: str, v2: str) -> bool:
        """Check if two values match using appropriate comparator."""
        # Empty check
        if not v1 and not v2:
            return True
        if not v1 or not v2:
            return False
        
        # Try numeric comparison first
        if self.numeric_comparator.compare(v1, v2):
            return True
        
        # Try unit comparison
        if self.unit_comparator.compare(v1, v2):
            return True
        
        # Fall back to text comparison
        return self.text_comparator.compare(v1, v2)
    
    def _classify_conflict(self, field: str, v1: str, v2: str) -> str:
        """Classify the type of conflict."""
        field_lower = field.lower()
        
        # Check if it's a unit field
        if "unit" in field_lower:
            return "unit"
        
        # Check if values are numeric
        if self.numeric_comparator._extract_number(v1) is not None:
            return "numeric"
        
        # Default to text
        return "text"
    
    def _assess_severity(
        self, 
        field: str, 
        v1: str, 
        v2: str,
        row_context: Optional[Dict] = None
    ) -> ConflictSeverity:
        """Assess the severity of a conflict."""
        field_lower = field.lower()
        
        # Extract the actual field name from "row[i].field"
        if "." in field_lower:
            field_lower = field_lower.split(".")[-1]
        
        # Check if it's a critical parameter by field name
        for critical in self.critical_params:
            if critical in field_lower:
                return ConflictSeverity.CRITICAL
        
        # Check if row context contains critical parameter name
        if row_context:
            param_name = str(row_context.get("Parameter", "")).lower()
            for critical in self.critical_params:
                if critical in param_name:
                    return ConflictSeverity.CRITICAL
        
        # Check numeric difference magnitude
        n1 = self.numeric_comparator._extract_number(v1)
        n2 = self.numeric_comparator._extract_number(v2)
        
        if n1 is not None and n2 is not None:
            max_val = max(abs(n1), abs(n2))
            if max_val > 0:
                diff_pct = abs(n1 - n2) / max_val
                if diff_pct > 0.1:  # >10% difference
                    return ConflictSeverity.HIGH
                elif diff_pct > 0.05:  # >5% difference
                    return ConflictSeverity.MEDIUM
        
        return ConflictSeverity.LOW
    
    def _resolve_conflicts(
        self,
        vision_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        conflicts: List[ConflictDetail]
    ) -> Tuple[Dict[str, Any], List[ConflictDetail]]:
        """Resolve conflicts and return merged result."""
        # Start with Vision result as base (deep copy)
        import copy
        merged = copy.deepcopy(vision_result)
        
        resolved_conflicts = []
        
        for conflict in conflicts:
            # Use arbiter to decide winner
            winner = self.arbiter.arbitrate(conflict)
            
            # Update conflict with resolution
            conflict.resolution = winner
            
            if winner == "rule_wins":
                conflict.resolved_value = conflict.rule_value
                # Update merged result
                self._apply_resolution(merged, conflict)
            else:
                conflict.resolved_value = conflict.vision_value
            
            resolved_conflicts.append(conflict)
        
        return merged, resolved_conflicts
    
    def _apply_resolution(self, result: Dict[str, Any], conflict: ConflictDetail):
        """Apply a conflict resolution to the result."""
        field = conflict.field
        
        # Handle row fields: "row[i].key"
        match = re.match(r'row\[(\d+)\]\.(.+)', field)
        if match:
            row_idx = int(match.group(1))
            key = match.group(2)
            
            if "rows" in result and row_idx < len(result["rows"]):
                result["rows"][row_idx][key] = conflict.resolved_value
            return
        
        # Handle direct fields
        if field in result:
            result[field] = conflict.resolved_value
    
    def _calculate_confidence(
        self,
        conflicts: List[ConflictDetail],
        resolved_conflicts: List[ConflictDetail]
    ) -> float:
        """Calculate confidence score based on conflicts."""
        if not conflicts:
            return 1.0
        
        # Start with base confidence
        confidence = 1.0
        
        # Deduct for each conflict based on severity
        severity_penalties = {
            ConflictSeverity.LOW: 0.02,
            ConflictSeverity.MEDIUM: 0.05,
            ConflictSeverity.HIGH: 0.10,
            ConflictSeverity.CRITICAL: 0.20,
        }
        
        for conflict in conflicts:
            penalty = severity_penalties.get(conflict.severity, 0.05)
            
            # Reduce penalty if resolved
            if conflict.resolution:
                penalty *= 0.5
            
            confidence -= penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _has_large_difference(self, conflicts: List[ConflictDetail]) -> bool:
        """Check if any conflict has a large numeric difference (>10%)."""
        for c in conflicts:
            n1 = self.numeric_comparator._extract_number(c.vision_value)
            n2 = self.numeric_comparator._extract_number(c.rule_value)
            
            if n1 is not None and n2 is not None:
                max_val = max(abs(n1), abs(n2))
                if max_val > 0:
                    diff_pct = abs(n1 - n2) / max_val
                    if diff_pct > 0.10:  # >10% difference
                        return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset verification statistics."""
        for key in self._stats:
            self._stats[key] = 0


# ============================================================================
# Convenience Functions
# ============================================================================

def create_verifier(
    tolerance: float = 0.01,
    threshold: float = 0.60,
    critical_params: Optional[List[str]] = None
) -> DoubleVerifier:
    """Factory function to create a configured DoubleVerifier."""
    return DoubleVerifier(
        numeric_tolerance=tolerance,
        confidence_threshold=threshold,
        critical_params=critical_params
    )


def quick_verify(
    vision_result: Dict[str, Any],
    rule_result: Optional[Dict[str, Any]] = None
) -> Tuple[str, float]:
    """
    Quick verification returning just status and confidence.
    
    Returns:
        Tuple of (status_string, confidence_float)
    """
    verifier = DoubleVerifier()
    result = verifier.verify(vision_result, rule_result)
    return result.status.value, result.confidence


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test the verifier
    vision_table = {
        "table_id": "test_1",
        "title": "Electrical Characteristics",
        "headers": ["Parameter", "Min", "Typ", "Max", "Unit"],
        "rows": [
            {"Parameter": "VIN Range", "Min": "4.2", "Typ": "", "Max": "36", "Unit": "V"},
            {"Parameter": "IQ", "Min": "", "Typ": "25", "Max": "40", "Unit": "µA"},
        ]
    }
    
    rule_table = {
        "table_id": "test_1",
        "title": "Electrical Characteristics",
        "headers": ["Parameter", "Min", "Typ", "Max", "Unit"],
        "rows": [
            {"Parameter": "VIN Range", "Min": "4.2", "Typ": "", "Max": "36", "Unit": "V"},
            {"Parameter": "IQ", "Min": "", "Typ": "25", "Max": "40", "Unit": "uA"},  # Different unit format
        ]
    }
    
    verifier = DoubleVerifier()
    result = verifier.verify(vision_table, rule_table)
    
    print(f"Status: {result.status.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Resolution: {result.resolution_method}")
    print(f"Conflicts: {len(result.conflicts)}")
    
    for conflict in result.conflicts:
        print(f"  - {conflict.field}: '{conflict.vision_value}' vs '{conflict.rule_value}' "
              f"[{conflict.severity.value}] -> {conflict.resolution}")
