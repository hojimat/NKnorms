"""
Categorizes simulation results by the given
directory structure and naming patterns.

Needs accompanying structure.json file.
"""

import sys
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

def categorize(dirs, patterns):
    """Move files to the subdirectories by matching patterns"""

    # first element of dirs in json is always root or top dir
    for top_dir in dirs[0]:
        # create a path object
        top_dir = Path(top_dir)

        # get the flat list of all possible directory names
        categories = [z for _ in dirs[1:] for z in _]

        # sort files into categories by patterns
        for file in top_dir.rglob('*'):
            # ignore directories
            if not file.is_file():
                continue

            # build path from scratch
            dest_path = top_dir
            for cat in categories:
                if re.search(patterns[cat], file.name):
                    dest_path /= cat
            if dest_path != top_dir:
                file.rename(dest_path / file.name)

def flatten(dirs):
    """Moves all files back to the root directory and removes empty directories"""
    for root_dir in dirs[0]:
        root_dir = Path(root_dir)
        for file in root_dir.rglob('*'):
            if file.is_file():
                file.rename(root_dir / file.name)

    for path in itertools.product(*dirs[:2]):
        p = Path(*path)
        shutil.rmtree(p)

def main():
    """Main function"""

    if len(sys.argv) < 2:
        sys.exit("No arguments provided.")
    action = sys.argv[1]

    with open('structure.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  
        structure = data['structure']
        patterns = data['patterns']

    if action == "--flatten":
        flatten(structure)
    elif action == "--categorize":
        scaffold(structure)
        categorize(structure, patterns)
    else:
        sys.exit("Incorrect arguments provided.")


if __name__=='__main__':
    main()
