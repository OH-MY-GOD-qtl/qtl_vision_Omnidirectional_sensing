#!/bin/bash

# 设置脚本运行的工作目录为脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "无法进入脚本目录: $SCRIPT_DIR"; exit 1; }

# 等待系统完全启动
sleep 5

# 使用screen启动watchdog脚本
screen -S qtl_vision \
       -L \
       -Logfile "./logs/screen_$(date "+%Y-%m-%d_%H-%M-%S").log" \
       -d \
       -m \
       bash -c "./watchdog.sh"

echo "qtl_vision watchdog已在screen会话中启动"
echo "使用 'screen -r qtl_vision' 查看运行状态"