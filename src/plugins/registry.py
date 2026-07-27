"""
Plugin registry — discovers and instantiates pipeline plugins from config.

Usage in project config JSON:
  "plugins": [
    { "type": "redcap", "source": "api", "api_url_env": "REDCAP_API_URL", ... },
    { "type": "redcap", "source": "csv", "codebook_path": "source_data/dict.csv" }
  ]

Adding a new plugin type:
  Register its class in PLUGIN_REGISTRY below.
"""

import os
import logging
logger = logging.getLogger(__name__)
from plugins.base import PipelinePlugin


def _resolve_env(value: str) -> str:
    """Expand a ``${VAR}`` reference to its environment value.

    Bare strings (and ``${VAR}`` for an unset VAR) are returned unchanged.
    """
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], value)
    return value


def _resolve_config_values(config: dict) -> dict:
    """Walk config and expand any string values that reference env vars."""
    resolved = {}
    for k, v in config.items():
        if isinstance(v, str):
            resolved[k] = _resolve_env(v)
        elif isinstance(v, dict):
            resolved[k] = _resolve_config_values(v)
        else:
            resolved[k] = v
    return resolved


def load_plugins(plugin_configs: list[dict]) -> list[PipelinePlugin]:
    """instantiate plugins from a list of config dicts"""
    from plugins.redcap.plugin import REDCapPlugin
    from plugins.provenance.plugin import ProvenancePlugin

    PLUGIN_REGISTRY: dict[str, type[PipelinePlugin]] = {
        "redcap": REDCapPlugin,
        "provenance": ProvenancePlugin,
    }

    plugins: list[PipelinePlugin] = []
    for cfg in plugin_configs or []:
        plugin_type = cfg.get("type", "").lower()
        cls = PLUGIN_REGISTRY.get(plugin_type)
        if cls is None:
            logger.warning("Unknown plugin type '%s' — skipping.", plugin_type)
            continue
        try:
            resolved_cfg = _resolve_config_values(cfg)
            plugin = cls(resolved_cfg)
            plugins.append(plugin)
            logger.info("Loaded plugin: %s", plugin.plugin_id)
        except Exception as e:
            logger.error("Failed to load plugin '%s': %s", plugin_type, e)
    return plugins
