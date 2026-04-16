#!/usr/bin/env python
"""Inspect notebook cell structure.

Utility script to examine the cells in the EDA notebook.
Run from project root: python scripts/inspect_notebook_cells.py
"""

import json

with open('notebooks/eda.ipynb', 'r') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("First 10 cell IDs:")
for i, cell in enumerate(nb['cells'][:10]):
    cell_id = cell.get('metadata', {}).get('id', 'no-id')
    cell_type = cell.get('cell_type', 'unknown')
    source = cell.get('source', [])
    first_line = source[0][:50] if source else '(empty)'
    print(f"{i+1}. ID: {cell_id}, Type: {cell_type}, Content: {first_line}")
