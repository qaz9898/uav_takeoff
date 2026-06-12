#!/bin/bash
# 树莓派 OS 禁止直接 pip3 install 到系统 Python，用项目虚拟环境安装依赖。
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
echo ""
echo "完成。之后请用虚拟环境里的 Python，例如："
echo "  .venv/bin/python3 src/run_takeoff.py"
