import re
import shutil
import json
import itertools
from pathlib import Path

def create(dirs):
    for path in itertools.product(*dirs):
        p = Path(*path)
        p.mkdir(parents=True, exist_ok=True)

def move(file_path, dest_dir):
    shutil.move(str(file_path), str(dest_dir / file_path.name))

def process(top_dir, patterns):
    top_dir = Path(top_dir)
    for file in top_dir.rglob('*'):
        if file.is_file():
            dest_path = top_dir

            for dir_name, pattern in patterns.items():
                if re.search(pattern, file.name):
                    dest_path /= dir_name
            
            if dest_path != top_dir:
                move(file, dest_path)

def main():
    with open("structure.json", "r") as file:
        data = json.load(file)

    structure = data['structure']
    patterns = data['patterns']

    create(structure)
    process('Xperf', patterns)
    


if __name__=='__main__':
    main()