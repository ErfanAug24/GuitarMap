import os
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional


class ConfigError(Exception):
    """
    Custom exception for configuration-related errors.
    """
    pass


class Config:
    """
    Lightweight hierarchical config loader
    """

    def __init__(self, yaml_path: str = None):
        self._config: Dict[str, Any] = {}

        if yaml_path:
            path = Path(yaml_path)
            if not path.exists():
                raise ConfigError(f"Config file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        self.merge_env_vars()

    def _merge_env_vars(self):
        """
        Merge in environment variables that match config keys (case-insensitive).
        :return:
        """
        for key, value in os.environ.items():
            normalized_key = key.lower().replace("__", ".")
            if normalized_key in self._flatten_keys(self._config):
                self._set_deep(self._config, normalized_key, value)

    def get(self, dotted_key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve nested keys using dotted path access e.g.,'audio.sample_rate'
        :param dotted_key:
        :param default:
        :return:
        """
        parts = dotted_key.split(".")
        ref = self._config
        for p in parts:
            if isinstance(ref, dict) and p in ref:
                ref = ref[p]
            else:
                return default
        return ref

    def _flatten_keys(self, d: Dict[str, Any], parent_key: str = "", sep=".") -> Dict[str, Any]:
        """
        Flatten nested keys into dotted form for env var lookup
        :param d:
        :param parent_key:
        :param sep:
        :return:
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_keys(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def _set_deep(d: Dict[str, Any], dotted_key: str, value: Any):
        """
        Set a nested key value using dotted path notation.
        :param d:
        :param dotted_key:
        :param value:
        :return:
        """
        parts = dotted_key.split(".")
        ref = d
        for p in parts[:-1]:
            ref = ref.setdefault(p, {})
        ref[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the full config dictionary.
        :return:
        """
        return self._config


@lru_cache()
def load_config(path: str = "app/config/defaults.yaml"):
    return Config(path)
