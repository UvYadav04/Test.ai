"""Ensures the repo root is importable as top-level packages (ingestion, vectordb)
regardless of where pytest is invoked from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
