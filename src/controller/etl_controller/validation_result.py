from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ValidationResult:
    """Result of validating a single FHIR resource against a profile."""

    is_valid: bool
    resource_type: Optional[str] = None
    profile_url: Optional[str] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
