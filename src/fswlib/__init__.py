# src/fswlib/__init__.py
"""
.. include:: ../../README.md
"""

from .fsw_embedding.fsw_embedding import (FSWEmbedding,
                                          TotalMassEncodingTransformation,
                                          TotalMassEncodingMethod,
                                          EnumWithResolve,
                                          FrequencyInitMethod,
                                          FSWCustomCudaExtensionLoadWarning,
                                          FSWCustomCudaExtensionLoadError)

from .fsw_embedding.test_fsw_embedding import main as test_fsw_embedding

__all__ = ["FSWEmbedding", "FSWCustomCudaExtensionLoadWarning", "FSWCustomCudaExtensionLoadError", "EnumWithResolve", "TotalMassEncodingTransformation", "TotalMassEncodingMethod", "FrequencyInitMethod", "test_fsw_embedding"]