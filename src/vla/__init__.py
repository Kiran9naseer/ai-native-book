"""
VLA (Vision-Language-Action) Library

Core library for humanoid robot voice control and LLM-based action planning.
"""

from .llm_client import LLMClient, ActionPlan, ActionPrimitive, create_llm_client

__version__ = "0.1.0"

__all__ = [
    "LLMClient",
    "ActionPlan", 
    "ActionPrimitive",
    "create_llm_client",
]
