import pandas as pd

import json
# Read the JSON file
with open('/private/home/irenewang/HWNAS/phaze/Estimator/arch_configs/cores.json', 'r') as f:
    data = json.load(f)
# Create new entries with "L2_BW": 32 for each entry with "L2_BW": 64
new_data = []
for entry in data:
    if entry['L2_BW'] == 64:
        new_entry = entry.copy()
        new_entry['L2_BW'] = 32
        new_data.append(new_entry)
    new_data.append(entry)
# Write the result to a new JSON file
with open('new_cores.json', 'w') as f:
    json.dump(new_data, f, indent=4)