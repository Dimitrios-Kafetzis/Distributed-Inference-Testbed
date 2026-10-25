"""The Neural Network Architecture Partitioning (NNPM) manager: the web application.

NNPM is served from the edge server and accessed through any browser on the
management network. It renders the topology on an interactive map, probes the
reachability of every node, and captures the parameters of each experimental
step in modal dialogs (scenario creation, benchmarking, partition-point
optimisation) before dispatching it through the orchestration layer of
``src/orchestration``.
"""
