#!/usr/bin/env bash
# we follow https://matt.sh/python-project-structure-2024 and use poetry and pyroject.toml
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

# set python executable
PYTHON=python3.12

# create and activate virtual environment
$PYTHON -m venv .venv
source $DIR/.venv/bin/activate

# install poetry and basic utilities
pip install pip poetry wheel setuptools -U

# set keyring backend to false to avoid poetry keyring issues
# https://github.com/python-poetry/poetry/issues/8623#issuecomment-2013653020
poetry config keyring.enabled false

# install dependencies from pyproject.toml
poetry install --no-root
