"""TensorFlow (Keras) model adapters for the DNN experiments.

The environment defaults below must be set before TensorFlow is imported by
any adapter module: CPU only, and no TensorFlow runtime messages in the log (level 3:
on a machine without a GPU, TensorFlow 2.15 prints spurious error-level messages about
CUDA plug-in registration at import, which would fill the merged experiment log; real
failures still surface as Python exceptions).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
