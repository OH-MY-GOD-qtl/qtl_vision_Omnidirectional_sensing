#!/bin/bash

# 设置脚本运行的工作目录为脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "无法进入脚本目录: $SCRIPT_DIR"; exit 1; }


# 配置参数 
# APP_NAME="standard_mpc"                # standard_mpc
APP_NAME="auto_aim_debug_mpc"          # auto_aim_debug_mpc
APP_PATH="./build/$APP_NAME"           # 应用程序路径
CONFIG_FILE="./configs/standard3.yaml"  # 配置文件路径
MAX_RETRY=100                          # 最大重试次数
CHECK_INTERVAL=5                        # 检查间隔（秒）
RESTART_DELAY=1                         # 重启延迟（秒）

# 定义退出函数和信号捕获
cleanup() {
  echo "检测到终止信号，退出脚本..."
  pkill -P $$  # 杀死所有子进程
  exit 1
}
trap cleanup SIGINT SIGTERM

while true; do
  # 检查 $APP_NAME 进程是否存在
  if ! pidof $APP_NAME > /dev/null; then
    echo "$APP_NAME 未运行，正在重启..."

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ "$RETRY_COUNT" -gt "$MAX_RETRY" ]; then
      echo "错误: 进程崩溃重启已达上限 ($MAX_RETRY 次)"
      cleanup
    fi

    sleep $RESTART_DELAY

    # 启动程序（后台运行）
    $APP_PATH $CONFIG_FILE 2>&1 &

  else
    # 进程运行正常时重置崩溃计数器
    RETRY_COUNT=0
  fi

  sleep 5
done