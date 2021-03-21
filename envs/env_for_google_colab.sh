#!/bin/bash
git clone https://github.com/NVIDIA/apex.git apex && cd apex && python3 setup.py install

pip3 install -q torch_snippets
pip3 install -q pyyaml>=5.1.1
pip3 install -q coolname>=1.1.0
pip3 install -q tabulate>=0.8.3
pip3 install -q tensorboardX>=1.4
pip3 install -q runx==0.0.6