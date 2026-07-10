"""Shared, agent-agnostic building blocks (duplicated per agent by choice)."""

from .llm import build_llm

__all__ = ["build_llm"]
