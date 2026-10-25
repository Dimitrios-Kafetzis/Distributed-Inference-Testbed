"""The inference scripts: model adapters, FLOP counting and per-layer profiling.

A model adapter exposes a model as an ordered list of stages (layers or
blocks) that can be executed one at a time or as a contiguous range, so that
the same code profiles any model layer by layer and runs it partitioned at
any point. Adapters are registered by name (``src/models/registry.py``);
``src/models/inference.py`` is the device-side entry point.
"""
