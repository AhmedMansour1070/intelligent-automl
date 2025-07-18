"""
Models module for the Intelligent AutoML framework.

This module provides automatic model training and selection capabilities.
"""

try:
    from .auto_trainer import AutoModelTrainer
    __all__ = ["AutoModelTrainer"]
except ImportError:
    # Auto trainer not fully implemented yet, but that's okay
    __all__ = []

# Module metadata
__version__ = "1.0.0"
__author__ = "Intelligent AutoML Team"