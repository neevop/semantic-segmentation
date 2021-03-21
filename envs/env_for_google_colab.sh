#!/bin/bash
git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext

pip install -q torch_snippets pytorch_model_summary
