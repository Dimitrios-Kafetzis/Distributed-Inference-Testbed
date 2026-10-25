"""Distributed Inference Testbed: source packages.

Layout (Appendix B of the thesis):
  src/network/       networking and SSH utilities
  src/benchmarking/  the profiling pipeline
  src/partitioning/  the partition-optimisation module
  src/testbed/       the load-injection mechanisms
  src/models/        the inference scripts
Additional packages:
  src/common/        shared infrastructure (command runners, clocks, run journal)
  src/inference/     partitioned inference over the network
  src/orchestration/ experiment pipeline shared by NNPM and the command line
"""
