#!/usr/bin/env python
# coding: utf-8

# run this script to download the pretrained gptcast models from Zenodo
import sys
from pathlib import Path
project_path = Path(__file__).parent.parent.resolve()
data_path = project_path / "models"

sys.path.append(str(project_path))
from gptcast.utils.downloads import download_all_pretrained_models
from fire import Fire

def main(path: str = None, overwrite: bool = False):
    path = path or str(data_path)
    assert download_all_pretrained_models(Path(path), overwrite), "Failed to download one or more models"

if __name__ == '__main__':
    Fire(main)
