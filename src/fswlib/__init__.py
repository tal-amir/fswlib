# src/fswlib/__init__.py
from .fsw_embedding.fsw_embedding import FSWEmbedding, FSWCustomCudaExtensionLoadWarning, FSWCustomCudaExtensionLoadError
from .fsw_embedding.test_fsw_embedding import main as test_fsw_embedding

__all__ = ["FSWEmbedding", "FSWCustomCudaExtensionLoadWarning", "FSWCustomCudaExtensionLoadError", "test_fsw_embedding"]