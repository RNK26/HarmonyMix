"""
conftest.py — pytest session-level fixtures shared across all test files.
"""
import os
import sys

# Make sure the project root is on sys.path so `src.*` imports resolve
# whether pytest is invoked from the root or from inside the tests/ folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
