#!/usr/bin/env bash

set -e

echo "Installing system dependencies..."

sudo apt update

sudo apt install -y \
    build-essential \
    gcc \
    make \
    wget \
    curl \
    python3-dev \
    pkg-config

cd /tmp

echo "Downloading TA-Lib..."

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -xzf ta-lib-0.4.0-src.tar.gz

cd ta-lib

echo "Compiling and installing TA-Lib"

./configure --prefix=/usr

make

sudo make install

echo "Installation completed successfully."