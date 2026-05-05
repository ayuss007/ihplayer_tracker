from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RuntimeConfig:
    data: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data=data)

    def get(self, *keys: str, default=None):
        value = self.data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value
