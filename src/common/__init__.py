"""Shared infrastructure: command runners, clocks, sysfs access, seeds, runs, nodes.

Everything that touches the operating system, the network or the clock goes
through an interface defined here, so that every module can be tested without
hardware by injecting a fake.
"""
