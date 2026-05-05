#!/bin/bash

set -e

echo "=== 金融风控实验环境搭建 ==="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "错误: 需要 Python 3.9 或更高版本，当前版本: $python_version"
    exit 1
fi
echo "✓ Python版本检查通过: $python_version"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建虚拟环境
if [ -d "venv" ]; then
    echo "检测到已存在虚拟环境，正在删除..."
    rm -rf venv
fi
echo "正在创建虚拟环境..."
python3 -m venv venv
echo "✓ 虚拟环境创建完成"

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo "正在升级pip..."
pip install --upgrade pip

# 安装依赖
echo "正在安装核心依赖..."
pip install -r requirements.txt

echo ""
echo "=== 环境搭建完成 ==="
echo "激活虚拟环境: source venv/bin/activate"
echo "运行实验: python your_experiment.py"
