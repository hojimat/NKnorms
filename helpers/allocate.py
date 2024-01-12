"""
Categorizes simulation results by the given
directory structure and naming patterns.

Needs accompanying structure.json file.
"""

import re
import shutil
import json
import itertools
from pathlib import Path

def scaffold(dirs):
    """Create directories from structure.json"""

    for path in itertools.product(*dirs):
        p = Path(*path)
        p.mkdir(parents=True, exist_ok=True)

def categorize(top_dir, patterns):
    """Move files to the subdirectories by matching patterns"""

    top_dir = Path(top_dir)
    for file in top_dir.rglob('*'):
        if file.is_file():
            dest_path = top_dir

            for dir_name, pattern in patterns.items():
                if re.search(pattern, file.name):
                    dest_path /= dir_name

            if dest_path != top_dir:
                file.rename(dest_path / file.name)

def flatten(root_dir):
    """Moves all files back to the root directory and removes empty directories"""

    root_dir = Path(root_dir)
    for file in root_dir.rglob('*'):
        if file.is_file():
            file.rename(root_dir / file.name)

def main():
    """Main function"""

    with open('structure.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  
        structure = data['structure']
        patterns = data['patterns']

    scaffold(structure)
    categorize('perf', patterns)
    #flatten('sync')


if __name__=='__main__':
    main()
