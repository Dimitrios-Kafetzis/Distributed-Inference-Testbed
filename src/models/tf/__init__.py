"""TensorFlow (Keras) model adapters for the DNN experiments.

The environment defaults below must be set before TensorFlow is imported by
any adapter module: CPU only, and only warnings and errors in the log.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
