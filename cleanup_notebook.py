#!/usr/bin/env python
import json

# Load the notebook
with open('notebooks/eda.ipynb', 'r') as f:
    nb = json.load(f)

print(f"Original notebook: {len(nb['cells'])} cells")

# Keep cells 0-21 (first 22 cells) and 35-38 (last 4 cells, originally 36-39)
# This deletes cells 23-35 (indices 22-34)
nb['cells'] = nb['cells'][:22] + nb['cells'][35:]

print(f"After cleanup: {len(nb['cells'])} cells")

# Save the notebook
with open('notebooks/eda.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("\nFinal cell structure:")
for idx, cell in enumerate(nb['cells'], 1):
    cell_id = cell.get('id', 'no-id')
    cell_type = cell['cell_type']
    if cell_type == 'code':
        source = ''.join(cell['source']).split('\n')[0][:55]
        print(f"{idx:2d}. [{cell_id}] code: {source}")
    else:
        source = ''.join(cell['source']).split('\n')[0][:55]
        print(f"{idx:2d}. [{cell_id}] markdown: {source}")
