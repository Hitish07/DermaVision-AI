import os
import glob

def count_loc_and_files(directory, extensions):
    total_loc = 0
    total_files = 0
    for root, _, files in os.walk(directory):
        if 'node_modules' in root or '.git' in root or '__pycache__' in root or 'dist' in root:
            continue
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                total_files += 1
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        total_loc += len(f.readlines())
                except:
                    pass
    return total_files, total_loc

py_files, py_loc = count_loc_and_files('.', ['.py'])
ts_files, ts_loc = count_loc_and_files('.', ['.ts', '.tsx'])
css_files, css_loc = count_loc_and_files('.', ['.css'])

print(f"Python Files: {py_files} (LOC: {py_loc})")
print(f"TypeScript Files: {ts_files} (LOC: {ts_loc})")
print(f"CSS Files: {css_files} (LOC: {css_loc})")
