#!/bin/bash
set -e

# Function to handle errors
error_handler() {
    echo "Error occurred in script at line: $1"
    exit 1
}

# Trap errors and call error_handler
trap 'error_handler $LINENO' ERR

# install python3.12
sudo apt update && \
sudo apt install software-properties-common -y && \
sudo add-apt-repository ppa:deadsnakes/ppa -y && \
sudo apt update && \
sudo apt install python3.12 python3.12-dev python3.12-venv -y
