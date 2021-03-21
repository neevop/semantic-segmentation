#!/bin/bash
git clone https://github.com/NVIDIA/apex.git apex && cd apex && python3 setup.py install

pip install -q torch_snippets
pip install -q pyyaml
pip install -q coolname
pip install -q tabulate
pip install -q tensorboardX
pip install -q runx