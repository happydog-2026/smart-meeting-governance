#!/usr/bin/env bash
# Mac / Linux 启动脚本(Windows 请双击 start_web.bat)
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "未找到 Python3,请先安装 Python 3.11+(https://www.python.org/downloads/)"
  exit 1
fi

exec "$PY" run.py
