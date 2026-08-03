"""Bundle discovery and normalisation for the dashboard.

The project has produced results in two different on-disk shapes, and the
dashboard has to render either without knowing in advance which it is
looking at. This package hides that difference behind one schema.
"""

from __future__ import annotations

from src.bundles.registry import (
    BundleNotFound,
    describe_bundle,
    list_bundles,
    load_bundle,
    resolve_bundle_id,
)

__all__ = [
    "BundleNotFound",
    "describe_bundle",
    "list_bundles",
    "load_bundle",
    "resolve_bundle_id",
]
