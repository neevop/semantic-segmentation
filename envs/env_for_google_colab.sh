#!/bin/bash
git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext

pip install -q torch_snippets
pip install -q pytorch_model_summary
pip install -q pyyaml>=5.1.1
pip install -q coolname>=1.1.0
pip install -q tabulate>=0.8.3
pip install -q tensorboardX>=1.4
pip install -q runx==0.0.6