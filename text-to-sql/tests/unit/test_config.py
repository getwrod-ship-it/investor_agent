"""
memory_window_size configuration: default and env override.
"""

from app.config import get_settings


def test_memory_window_size_default():
    assert get_settings().memory_window_size == 5


def test_memory_window_size_env_override(monkeypatch):
    monkeypatch.setenv("MEMORY_WINDOW_SIZE", "3")
    get_settings.cache_clear()
    try:
        assert get_settings().memory_window_size == 3
    finally:
        # re-read with the original env for any test that runs later
        get_settings.cache_clear()
