"""
Minimal setup.py for legacy pip compatibility.

All build configuration lives in pyproject.toml.
This file exists only so that older pip versions (pre-21.3)
and tools like `pip install -e .` that still look for setup.py
can find the package metadata.

For modern builds, pyproject.toml is the source of truth:
  pip install .
  pip install -e .
  pip install .[torch,dev]
"""

from setuptools import setup

setup()
