#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  read -r -s -p 'Tushare token (hidden): ' TUSHARE_TOKEN
  printf '\n'
fi
export TUSHARE_TOKEN
# 转发地址属于本机配置：从环境变量读取，不写进仓库。
if [[ -z "${TUSHARE_BASE_URL:-}" ]]; then
  echo "请先设置 TUSHARE_BASE_URL（Tushare 接口或转发地址）" >&2
  exit 1
fi
export TUSHARE_BASE_URL
trap 'unset TUSHARE_TOKEN' EXIT
default_sdk="../ashare-data-pipeline/.venv/bin/python"
[ -x "$default_sdk" ] || default_sdk="../a-share-quant-agent/.venv/bin/python"  # 旧目录名
sdk_python="${TUSHARE_PYTHON:-$default_sdk}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$sdk_python" -m quant_research_lab.tushare_source archive \
  --transport sdk --allow-http --request-timeout 60 "$@"
