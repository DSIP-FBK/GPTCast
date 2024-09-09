#!/usr/bin/env python
# coding: utf-8

# run this script to download the miarad dataset from Zenodo
import sys
from pathlib import Path
project_path = Path(__file__).parent.parent.resolve()
data_path = project_path / "data"

sys.path.append(str(project_path))
from gptcast.utils.downloads import download_all_datasets
from fire import Fire

def main(path: str = None, overwrite: bool = False):
    path = path or str(data_path)
    assert download_all_datasets(Path(path), overwrite), "Failed to download one or more datasets"

if __name__ == '__main__':
    Fire(main)
