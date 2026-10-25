"""PyTorch model adapters for the Transformer-based LLM experiments.

Decoder-only models (GPT-2, TinyLlama) are exposed in two views: a stage view
at decoder-layer granularity (embedding, layer 1..L, language-model head),
which the profiler, ``inference.py`` and the partitioned-inference service use
unchanged, and a block view per layer (one unit per attention head, the
output projection, the feed-forward block) that the hook-based profiler times
with high-resolution timers. Everything runs on the CPU.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
