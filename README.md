# qtl_vision_Omnidirectional_sensing

RoboMaster 视觉系统：**自瞄**（传统视觉 / YOLO v5·v8·v11 + EKF 跟踪 + MPC 规划）、
**能量机关**（yolo11_buff）、**全向感知**、相机标定工具链与测试套件。
算法源自 dx_vision（同济 SuperPower 25 赛季开源），按"功能等价、算法优化、模块化清晰"标准重构。

---

## 目录

- [快速上手（海康相机 + 自瞄）](#快速上手海康相机--自瞄)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [编译](#编译)
- [程序入口速查](#程序入口速查)
- [配置说明](#配置说明)
- [全向感知](#全向感知)
- [模型与推理后端](#模型与推理后端)
- [标定流程](#标定流程)
- [通讯与串口协议](#通讯与串口协议)
- [部署自启动](#部署自启动可选)
- [测试与冒烟](#测试与冒烟)
- [常见问题 FAQ](#常见问题-faq)
- [与 dx_vision 的关系](#与-dx_vision-的关系)

---

## 快速上手（海康相机 + 自瞄）

```bash
cd qtl_vision_Omnidirectional_sensing  # 进入项目根目录

# ① 验证相机出图（-d 显示画面，窗口按 q 退出）
./build/camera_test configs/standard1.yaml -d

# ② 无云台跑自瞄全链路（模拟模式，不打开串口，只打印模拟下发）
./build/traditional_debug configs/standard1_sim.yaml          # 传统视觉检测
./build/auto_aim_debug_mpc configs/standard1_sim.yaml         # YOLO + MPC 调试版

# ③ 接上云台串口后的实机自瞄
./build/traditional_debug configs/standard1.yaml
./build/auto_aim_debug_mpc configs/standard1.yaml
```

前提：
- 海康相机已插入并被系统识别：`lsusb | grep 2bdf` 应有 `Hikrobot MV-...`
- 相机 USB 权限（udev 规则）已配置，见[环境准备](#环境准备)
- `standard1.yaml` 的相机段指向海康（`camera_name: "hikrobot"`、`vid_pid: "2bdf:0001"`）
- 实机自瞄需要云台串口 `/dev/ttyACM0` 存在，否则程序会直接退出（见 FAQ）

窗口退出统一按 **q**（或 Ctrl+C）；**不要按 Ctrl+Z**（进程只是被暂停，会继续占用相机/串口）。

---

## 项目结构

```
qtl_vision_Omnidirectional_sensing/
├── CMakeLists.txt              # 扁平构建：每模块一个 OBJECT 库，按注释分组
├── include/<模块>/xxx.hpp      # 头文件，按模块分目录
├── src/<模块>/xxx.cpp          # 源文件，与 include/ 镜像
├── apps/                       # 主程序入口（7 个）
├── calibration/                # 标定工具（5 个）
├── tests/                      # 测试（22 个，其中 3 个 ROS2 默认不编译）
├── smoke/                      # 冒烟工具（yolo_smoke）
├── configs/                    # yaml 配置
├── assets/                     # 模型文件（onnx/xml/bin/engine）+ demo 视频
├── third_party/
│   ├── serial/                 # 串口库（wjwwood/serial）
│   ├── tinympc/                # MPC 规划库（vendored）
│   └── camera_sdk/             # 海康/迈德威视 相机 SDK（头文件+库）
├── watchdog.sh autostart.sh    # 部署自启动脚本（screen 会话名 qtl_vision）
└── logs/ patterns/ records/    # 运行时生成（gitignore）
```

| 模块 | 职责 |
|---|---|
| `common` | 全局常量、Color 枚举、Mat33 旋转矩阵 |
| `image` | 图像预处理：灰度/二值/膨胀、letterbox、分离颜色通道双阈值 |
| `lightbar` | 灯条模型与算法：几何量计算、轮廓提取、几何筛选、颜色判定、PCA 角点修正 |
| `tools` | EKF、PID、弹道、数学工具、绘图、录像、CRC、日志、yaml 封装 |
| `camera` | 相机抽象：海康/迈德威视/USB/视频文件 |
| `comm` | 串口云台、裁判系统 cboard（CAN）、达妙 IMU、socketcan（ros2 预留） |
| `armor` | 装甲板数据模型（大小板类型、名称、优先级） |
| `classifier` | ONNX 数字分类器 |
| `detector` | 检测器：传统视觉 + YOLO v5/v8/v11 + 多线程检测 |
| `solve` | PnP 位姿解算 + 坐标变换 + yaw 重投影优化 |
| `tracker` | 11 维 EKF 目标模型 + 跟踪状态机 |
| `planner` | MPC 规划器（tinympc） |
| `fire` | 弹道迭代瞄准 + 开火判定 |
| `buff` | 能量机关：检测/解算/预测/瞄准（yolo11_buff） |
| `omni` | 全向感知：3 USB 传统检测 + 1 海康 YOLO，共 yaw、海康独享 pitch，海康优先决策 |

---

## 环境准备

```bash
sudo apt install -y libopencv-dev libyaml-cpp-dev libfmt-dev libeigen3-dev \
    libspdlog-dev nlohmann-json3-dev libusb-1.0-0-dev
# OpenVINO：apt 安装 openvino-2024.6.0（CMake 自动探测 /opt/intel 与 /usr/lib/cmake）
```

**TensorRT + CUDA**（本机 NVIDIA 显卡推理，CMake 构建必需，缺失则配置阶段报错）：

- 安装 NVIDIA 驱动、CUDA toolkit（`cuda_runtime_api.h` 与 `libcudart`）与 TensorRT
  （`NvInfer.h` 与 `libnvinfer`）。
- CMake 会自动探测头文件/库路径，默认 Hint 为
  `/usr/local/cuda`、`/usr/include/x86_64-linux-gnu`、`/usr/local/tensorrt`、`/opt/TensorRT`；
  自定义安装位置可手动指定 `TENSORRT_INCLUDE_DIR` / `TENSORRT_LIBRARY` /
  `CUDA_INCLUDE_DIR` / `CUDA_RUNTIME_LIBRARY`。
- 机器人端 CPU 部署若无需 TensorRT，当前 CMake 仍将其作为硬依赖；如需裁剪可
  在 `CMakeLists.txt` 中把 `find_*` 的 `FATAL_ERROR` 改为可选并去掉
  `tensorrt_infer.cpp` 的参与（详见[模型与推理后端](#模型与推理后端)）。

**海康相机 USB 权限**（必须，否则普通用户打不开相机，日志刷 `Unable to open usb!`）：

```bash
sudo sh -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"2bdf\", MODE=\"0666\"" > /etc/udev/rules.d/60-hikrobot.rules'
sudo udevadm control --reload && sudo udevadm trigger
```

**相机 SDK 说明**（重要）：

- SDK 位于 `third_party/camera_sdk/<厂商>/lib/{amd64,arm64}`。
- 海康 MVS 的核心库 `libMvCameraControl.so` 运行时会去**同目录**加载传输层库
  `libMvUsb3vTL.so`。当前 amd64 目录已放入 **MVS 5.0.2（SDK 4.8.1.2）完整库集**，
  缺其中任何一个 .so 都会导致相机枚举失败（`MV_CC_EnumDevices failed: 0x80000006`）。
- 程序已通过 rpath 链接该目录，**不需要**设置 `LD_LIBRARY_PATH`。
- `lib/arm64/`（机器人 Jetson 用）仍是旧版且**不完整**：上机前需用
  MVS 官方 aarch64 包按同样方式补齐，否则会出现与 x86 相同的枚举失败。
- 迈德威视目录仅有 `libMVSDK.so`。

ROS2 非编译必需（通讯为 ros2 预留，对应源码未参与构建）。

---

## 编译

```bash
mkdir build && cd build
cmake .. && make -j
```

生成 32 个可执行文件：主程序入口 7 个、标定工具 5 个、测试 19 个、冒烟工具 1 个。

> 构建同时依赖 OpenVINO 与 TensorRT/CUDA（见[环境准备](#环境准备)）；三者任一缺失
> 都会在 `cmake` 配置阶段报错。TensorRT 仅 YOLOv5 链路用到，其余模型走 OpenVINO。

---

## 程序入口速查

| 程序 | 源文件 | 用途 | 检测 | 瞄准/规划 | 下发通道 | 默认/推荐配置 |
|---|---|---|---|---|---|---|
| `camera_test` | `tests/camera_test.cpp` | 相机出图+fps 验证（`-d` 显示画面） | — | — | — | `standard1.yaml` |
| `MAIN` | `apps/main.cpp` | 精简传统流水线（调试窗口） | 传统 | 弹道 Aimer+Shooter | 串口 Gimbal | `standard1.yaml`（无硬件用 `video_demo.yaml`） |
| `traditional_debug` | `apps/traditional_debug.cpp` | 传统视觉自瞄（调试窗口） | 传统 | MPC Planner | 串口 Gimbal | `standard1(_sim).yaml` |
| `auto_aim_debug_mpc` | `apps/auto_aim_debug_mpc.cpp` | YOLO+EKF+MPC 自瞄（重投影窗口） | YOLO | MPC Planner | 串口 Gimbal | `standard1(_sim).yaml` |
| `standard` | `apps/standard.cpp` | YOLO 自瞄（无窗口） | YOLO | 弹道 Aimer+Shooter | 裁判系统 CBoard（CAN） | `standard3.yaml` |
| `standard_mpc` | `apps/standard_mpc.cpp` | YOLO+MPC 自瞄（无窗口） | YOLO | MPC Planner | 串口 Gimbal | `standard3.yaml` |
| `mt_standard` | `apps/mt_standard.cpp` | 多线程检测版自瞄（无窗口） | YOLO（多线程） | 弹道 Aimer+Shooter | 串口 Gimbal | 显式指定配置 |
| `mt_auto_aim_debug` | `apps/mt_auto_aim_debug.cpp` | 多线程检测版自瞄（重投影窗口） | YOLO（多线程） | 弹道 Aimer+Shooter | 串口 Gimbal | 显式指定配置 |
| `camera_auto_aim_test` | `tests/camera_auto_aim_test.cpp` | 相机+自瞄离线测试（无通讯） | YOLO | 弹道 Aimer | — | `camera_auto_aim.yaml` |
| `capture` | `calibration/capture.cpp` | 采集标定图片（s 保存 / q 退出） | — | — | 串口 Gimbal | `calibration(_sim).yaml` |
| `calibrate_camera` | `calibration/calibrate_camera.cpp` | 计算内参（结果**只打印不写文件**） | — | — | — | `calibration(_sim).yaml` |
| `calibrate_handeye` | `calibration/calibrate_handeye.cpp` | 手眼标定（需真实云台姿态） | — | — | — | `calibration.yaml` |
| `calibrate_robotworld_handeye` | `calibration/calibrate_robotworld_handeye.cpp` | 手眼标定（机器人基座系） | — | — | — | `calibration.yaml` |
| `split_video` | `calibration/split_video.cpp` | 视频切帧 | — | — | — | — |
| `yolo_smoke` | `smoke/yolo_smoke.cpp` | YOLO 模型加载+推理冒烟 | YOLO | — | — | `video_demo.yaml` |

除 `MAIN` 与 `split_video` 外，所有程序均以 yaml 配置路径作为第一个命令行参数
（`MAIN` 不传参数时默认 `configs/standard1.yaml`）；`split_video` 的参数为
`输入路径 起始帧 结束帧 输出路径`，其余标定类工具的额外参数见[标定流程](#标定流程)。

约定：`*_sim.yaml` 为模拟配置（`simulate: true`），不打开串口、固定自瞄模式、
只打印模拟下发，用于无云台硬件时跑通算法链路。

---

## 配置说明

配置文件均为 yaml，路径作为各程序的第一个参数传入。常用键：

```yaml
enemy_color: "blue"            # 敌方颜色：blue/red（打错颜色=检出不到灯条）
camera_name: "hikrobot"        # hikrobot / mindvision / usb / video
exposure_ms: 10                # 曝光（海康）
gain: 16                       # 增益（海康）
vid_pid: "2bdf:0001"           # 相机 VID:PID，lsusb 查看
yolo_name: yolov5              # yolov5/yolov8/yolo11
yolov5_model_path: assets/0526.onnx      # YOLOv5 OpenVINO 后端模型（深圳大学开源）
use_tensorrt: true             # YOLOv5 推理后端：true=TensorRT / false=OpenVINO
yolov5_engine_path: assets/yolov5.engine # TensorRT 引擎（use_tensorrt=true 时生效）
min_confidence: 0.8            # 检测置信度阈值
use_traditional: true          # 传统视觉二次矫正（仅 YOLO 链路）
use_roi: false                 # 是否启用 roi 区域
color_threshold: 40            # 传统检测颜色差阈值（红敌R-B / 蓝敌B-R）
brightness_threshold: 100      # 传统检测亮度阈值
yaw_offset / pitch_offset      # 云台固定偏角补偿（度）
distance_offsets: [[0.5,-1,-6], ...]  # 距离相关偏移补偿（可选）
auto_fire: true                # 是否由视觉控制开火
camera_matrix / distort_coeffs # 相机内参（见标定流程）
R_camera2gimbal / t_camera2gimbal  # 相机→云台外参（手眼标定结果）
com_port: "/dev/ttyACM0"       # 云台串口
baud_rate: 115200
simulate: true                 # 模拟模式（见上）
```

各配置文件当前相机指向：

| 文件 | 相机 | 用途 |
|---|---|---|
| `standard1.yaml` / `standard3.yaml` / `mvs.yaml` / `calibration.yaml` | hikrobot | 实机自瞄/标定主配置 |
| `standard1_sim.yaml` / `calibration_sim.yaml` | hikrobot | 模拟模式（本次接入海康相机时生成） |
| `camera_auto_aim.yaml` | hikrobot | 相机+自瞄离线测试 |
| `camera.yaml` / `hero.yaml` | hikrobot | 相机测试/英雄 |
| `example.yaml` | 迈德威视 | 示例 |
| `demo.yaml` | hikrobot | YOLO 回放测试 |
| `video_demo.yaml` | 视频文件 | 无硬件调试（自带 `simulate: true`） |

---

## 全向感知

> 状态：模块与配置已就绪，**尚未接线到任何程序入口**（USB 相机未到位，暂不启用）。

全向感知方案：**3 路 USB 相机 + 1 路海康相机**，4 相机共用 yaw 轴、海康独享 pitch 轴。

| 相机 | 检测路线 | 作用 |
|---|---|---|
| USB × 3 | 传统视觉（灯条几何匹配，**不识别编号**） | 全向搜索：只报「有无目标 + 方位」，引导 yaw 转向 |
| 海康 × 1 | YOLO（TensorRT） | 主瞄准：高精度自瞄，yaw/pitch 均由其驱动 |

决策逻辑（`Decider::decide(detection_queue)`）：

- **海康优先**：海康视野内有目标 → `Command{yaw, pitch}` 全由海康目标驱动；
- 海康无目标 → 用 USB 中优先级最高的目标驱动 yaw，**pitch 固定为预设值**；
- 都没有 → 输出 `control=false`。

配置键（见 `configs/omni.yaml`，USB 的 FOV 复用 `fov_h/fov_v`）：

```yaml
omni_usb_yaw_offsets: [60, -60, -180]  # 三个USB相对yaw零位的安装角(degree)
omni_hik_yaw_offset: 0                  # 海康相对yaw零位安装角(degree)
omni_hik_fov_h: 54.2                    # 海康水平FOV(degree)
omni_hik_fov_v: 44.5                    # 海康垂直FOV(degree)
omni_pitch_preset: 0                    # 无海康目标时pitch预设(degree)
```

相关代码：`omni/perceptron`（3 个传统线程 + 1 个 YOLO 线程并行采图检测）、
`omni/decider`（过滤 / 排序 / 海康优先决策）、`detector`（`use_classifier=false`
跳过数字分类器，仅灯条几何）、`camera/usbcamera`（按 sharpness 识别左右 left/right）。

---

## 模型与推理后端

推理引擎有两套：**OpenVINO**（机器人端 CPU 部署，默认）与 **TensorRT**
（WSL 本机 NVIDIA 显卡），YOLOv5 链路可通过配置一键切换。

### 模型清单

| 模型文件 | 用途 | 后端 | 说明 |
|---|---|---|---|
| `assets/0526.onnx` | YOLOv5 装甲板检测 | OpenVINO / TensorRT | 深圳大学开源模型，本项目默认自瞄检测模型 |
| `assets/yolov5.engine` | YOLOv5 装甲板检测 | TensorRT | 由 `0526.onnx` 转换的 TensorRT 序列化引擎 |
| `assets/yolov8.xml` | YOLOv8 装甲板检测 | OpenVINO | `yolo_name: yolov8` 时选用 |
| `assets/yolo11.xml` | YOLO11 装甲板检测 | OpenVINO | `yolo_name: yolo11` 时选用 |
| `assets/yolo11_buff_int8.xml` | 能量机关检测 | OpenVINO | INT8 量化，供 buff 模块 |
| `assets/tiny_resnet.onnx` | 装甲板数字分类器 | OpenVINO | 数字 1~5 识别 |

> 除 YOLOv5 外，其余模型（v8 / v11 / buff / 分类器）当前仅接入 OpenVINO 后端。

### 架构切换（YOLOv5）

切换由配置项控制，改 `use_tensorrt` 即可：

```yaml
use_tensorrt: true                      # true=TensorRT；false=OpenVINO
yolov5_model_path: assets/0526.onnx     # OpenVINO 后端加载的 ONNX
yolov5_engine_path: assets/yolov5.engine # TensorRT 后端加载的引擎
```

- `use_tensorrt: true`（本机 NVIDIA 显卡）：加载 `.engine` 引擎，
  经 `TensorRTInfer` 反序列化并推理。
- `use_tensorrt: false`（机器人端 CPU）：加载 `0526.onnx`，
  经 OpenVINO Runtime 编译推理（`device: CPU`）。

两种后端**共用同一套 letterbox 预处理与后处理解析**（
`src/detector/yolos/yolov5.cpp`），输入/输出语义一致，仅推理引擎不同。

### ONNX → TensorRT 引擎导出（trtexec）

`assets/yolov5.engine` 由 `0526.onnx` 通过 TensorRT 自带工具 `trtexec` 导出：

```bash
# 仅支持 FP32 精度（务必不要加 --fp16，见下方说明）
trtexec --onnx=assets/0526.onnx \--saveEngine=assets/yolov5.engine
```

> ⚠️ **只能用 FP32**：`--fp16` 导出的引擎在推理时会报错。
> 导出时**不要**加 `--fp16`、`--int8`、`--fp32`等精度参数。
> **AI**给的命令是带精度参数的是错的。


说明：

- `trtexec` 随 TensorRT 一起安装（deb 包通常在 `/usr/src/tensorrt/bin/trtexec`，
  也可直接调用系统 PATH 中的 `trtexec`）。
- 引擎文件与 **GPU 架构、TensorRT / CUDA 版本绑定**，换机器或升级版本后需重新
  导出，否则运行时报 `反序列化引擎失败`（见[常见问题 FAQ](#常见问题-faq)）。
- 本项目推理接口要求**静态形状、单输入**，`0526.onnx` 输入固定为 `1x3x640x640`，
  无需设置 `--minShapes / --optShapes / --maxShapes`。

### TensorRT 接口

`TensorRTInfer`（`include/detector/tensorrt_infer.hpp` +
`src/detector/tensorrt_infer.cpp`）是对 TensorRT C++ API 的 RAII 封装：

- 读取 `.engine` 序列化文件，复用 `IRuntime` / `ICudaEngine` / `IExecutionContext`
  与 CUDA 显存、主机缓冲（帧间零重复分配）；
- 当前约束：静态形状、单输入、**FP32**（引擎须以 FP32 导出，`--fp16` 会报错）；
- 推理接口 `infer(const float * input_nchw)` 接受主机 NCHW float32，返回主机侧
  输出张量（float32）。

---

## 标定流程

### 1. 相机内参（不需要串口）

1. 打印棋盘格标定板：内角点 **11×8**（12×9 个方格），方格边长 **20mm**
   （对应 `pattern_cols: 11, pattern_rows: 8, center_distance_mm: 20`），贴硬平板。
2. 采图（用模拟配置，避免无串口退出）：
   ```bash
   ./build/capture configs/calibration_sim.yaml assets/img_calib
   ```
   窗口内移动/倾斜标定板，覆盖画面四角、中心、近远，角点连线正确时按 **s**
   保存（15~25 张），按 **q** 退出。图片存为 `assets/img_calib/N.jpg`
   （同目录 `N.txt` 为模拟四元数，内参标定用不到）。
3. 计算内参：
   ```bash
   ./build/calibrate_camera configs/calibration_sim.yaml assets/img_calib
   ```
   逐张按任意键确认角点检测，最后**终端打印** `camera_matrix` 与
   `distort_coeffs`（**不会自动写入文件**），自行复制到运行配置
   （如 `standard1.yaml`）的对应键。
4. 检查打印的重投影误差，理想 < 0.5px；误差偏大则补拍多角度照片重算。

### 2. 手眼标定（必须接云台串口）

1. 接上云台串口，确认 `ls /dev/ttyACM0` 存在。
2. 用真实配置（不带 simulate）重新采图——每张图会记录真实云台姿态四元数：
   ```bash
   ./build/capture configs/calibration.yaml assets/img_with_q
   ```
3. 求解相机→云台外参：
   ```bash
   ./build/calibrate_handeye configs/calibration.yaml assets/img_with_q
   ```
4. 将结果填入运行配置的 `R_camera2gimbal` / `t_camera2gimbal`。

> 无串口时**无法**做手眼标定：模拟模式的四元数全是单位阵，解不出外参。

---

## 通讯与串口协议

- **串口云台**（`comm/gimbal`）：帧头 'S','P' + CRC16，与下位机约定保持一致，
  修改需两边同步；支持 `simulate` 模拟模式。
- **裁判系统 CBoard**（`comm/cboard`）：SocketCAN 读取裁判系统数据（imu/模式/弹速），
  仅 `standard` 依赖 can0（`standard_mpc` 使用串口 `Gimbal`）。
- **达妙 IMU**（`comm/dm_imu`）：串口读取 IMU 姿态。
- **ros2**（`comm/ros2`）：publish2nav/subscribe2nav 预留，未参与构建。

---

## 部署自启动（可选）

`autostart.sh` 启动 `watchdog.sh`（screen 会话名 `qtl_vision`）；watchdog 守护
`APP_NAME`（默认 `auto_aim_debug_mpc`）与 `CONFIG_FILE`（默认
`configs/standard1.yaml`），崩溃自动重启（上限 100 次）。部署时按需修改
`watchdog.sh` 内两个变量与路径，再按桌面自启动流程注册。

---

## 测试与冒烟

```bash
# YOLO 模型冒烟（加载+推理 300 帧，可用 video_demo 配置换模型验证 v8/v11）
./build/yolo_smoke configs/video_demo.yaml

# YOLO+EKF 视频回放（重投影画面；4 个参数：配置 起始帧 结束帧 输入前缀）
./build/auto_aim_test configs/demo.yaml 0 0 assets/demo
```

`assets/demo.avi`（约 60MB，gitignore）+ `assets/demo.txt`（每帧云台四元数真值，
620 帧）配套使用；缺失时从 dx_vision 仓库 `assets/demo/demo.avi` 复制。

其余 19 个 `*_test` 覆盖相机/检测/云台/弹道/规划等模块，ROS2 相关 3 个测试
默认不编译（CMake 中注释）。

---

## 常见问题 FAQ

| 现象 | 原因与处理 |
|---|---|
| `[Gimbal] Failed to open serial ... exit` 直接退出 | 云台串口未接或 `com_port` 不对。无硬件时用 `*_sim.yaml`（`simulate: true`），或先插好 `/dev/ttyACM0` |
| 相机日志刷 `MV_CC_EnumDevices failed: 0x80000006` | ① USB 权限：配 udev 规则或 sudo 跑；② SDK 库不全（缺 `libMvUsb3vTL.so`）；③ 相机被其他进程占用（`ps aux \| grep camera_test` 后 kill） |
| 相机日志刷 `Unable to open usb!` | USB 权限问题，配 udev 规则（见环境准备） |
| `Not found camera!` | 相机未插/未识别：`lsusb \| grep 2bdf` 确认；插拔后重试 |
| 进程退不掉 / 相机被占 | 按过 Ctrl+Z：进程只是被挂起。`ps aux` 找到 PID 后 `kill -9`，以后用 **q / Ctrl+C** 退出 |
| 画面花屏/卡死 | 相机 SDK 版本与头文件不匹配（如混用 3.x/4.8 库），保持同版本整套拷贝 |
| 检测不到装甲板 | ① `enemy_color` 与实际敌方颜色不符；② 曝光/增益不适配现场光照；③ 调 `color_threshold`/`brightness_threshold`；④ YOLO 检查 `min_confidence` |
| 自瞄弹道/距离明显偏 | 相机内参未更新（`camera_matrix` 仍是别的相机标定值），按标定流程重标 |
| `calibrate_camera` 没写文件 | 设计如此：结果只打印在终端，需手动复制进配置 |
| 想换相机型号 | 改配置 `camera_name` + `vid_pid`，确认对应 SDK 目录 lib 齐全 |
| 想在 CPU / GPU 间切换推理后端 | 改配置 `use_tensorrt`：`true` 走 TensorRT 引擎（`yolov5_engine_path`），`false` 走 OpenVINO（`yolov5_model_path`），仅 YOLOv5 支持 |
| `[TensorRT] 反序列化引擎失败` / `无法打开引擎文件` | ① `yolov5_engine_path` 指向的 `.engine` 不存在或与当前 TensorRT / GPU 架构不匹配；② 用同版本 TensorRT 重新导出引擎；③ 检查 CUDA 驱动与 `libnvinfer` 版本 |
| 机器人（arm64）相机打不开 | `third_party/camera_sdk/*/lib/arm64/` SDK 不完整，用官方 aarch64 包补齐 |

---

## 与 dx_vision 的关系

**标准：功能等价。** 算法逻辑与 dx 同源（检测→PnP→EKF→弹道/MPC→
下发全链路大体一致），工程层面重构：

1. **结构**：去除命名空间；`io/tools/tasks` 重组为 `include/<模块>/` +
   `src/<模块>/` 镜像；主程序收进 `apps/`；单文件扁平 CMake；从 detector/armor/buff
   拆出 `image`、`lightbar` 独立模块。
2. **链路优化**：预处理缓冲复用（消除每帧约 3MB 重复分配）、letterbox 去逐帧
   memset、修正 YOLO 二次矫正取色坐标系。
3. **预处理升级**：传统检测从灰度单阈值改为**分离颜色通道双阈值**
   （颜色差 + 亮度双掩码，默认 40/100），与桌面版 `PreProcess::process` 一致；
   注意 demo.avi 的过曝灯条在该阈值下检出率低，实机按需调整。
4. **有意差异**：新增 `usb`/`video` 相机源、`simulate` 串口模拟、GimbalState
   默认值；日志节流；资产平铺到 `assets/`；命名冲突改名；
   **YOLOv5 双后端推理**（新增 `TensorRTInfer` 接口，`use_tensorrt` 一键切换
   OpenVINO / TensorRT，见[模型与推理后端](#模型与推理后端)）。
5. **清理**：删除无人引用的 ceres 源码包、`archive/` 等残留与孤儿模型。
