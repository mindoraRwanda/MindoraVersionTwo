from typing import Optional
from functools import lru_cache
from .base import BaseAppSettings, get_environment
from .model import ModelSettings
from .performance import PerformanceSettings
from .safety import SafetySettings
from .cultural import CulturalSettings
from .database import DatabaseSettings
from .emotional import EmotionalResponseSettings
from .settings import Settings, get_settings

# Export all the main classes and functions
__all__ = [
    'BaseAppSettings',
    'get_environment',
    'ModelSettings',
    'PerformanceSettings',
    'SafetySettings',
    'CulturalSettings',
    'DatabaseSettings',
    'EmotionalResponseSettings',
    'Settings',
    'get_settings',
    'settings'
]

class _LazySettingsProxy:
    """Lightweight proxy that lazily loads the real Settings instance on first attribute access.

    This avoids heavy initialization at import time while preserving the `settings` symbol
    for backward compatibility. Accessing any attribute will instantiate and cache the real
    Settings object (via get_settings()).
    """
    __slots__ = ("_settings",)

    def __init__(self) -> None:
        self._settings = None

    def _load(self) -> None:
        if self._settings is None:
            self._settings = get_settings()

    def __getattr__(self, name: str):
        self._load()
        return getattr(self._settings, name)

    def __repr__(self) -> str:
        if self._settings is None:
            return "<LazySettingsProxy (unloaded)>"
        return repr(self._settings)

# Backwards-compatible `settings` object — lazy by default.
settings = _LazySettingsProxy()