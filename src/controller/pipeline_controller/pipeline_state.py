from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineState:
    """shared runtime state datat"""
    source_helper_urls: list = field(default_factory=list)
    structure_map_urls: list = field(default_factory=list)
    custom_mapping_table: Optional[dict] = None
    composition_descriptor: Optional[dict] = None
