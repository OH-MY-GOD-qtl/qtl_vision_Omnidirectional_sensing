cmake_minimum_required(VERSION 3.16)

project(qtl_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译优化：默认 Release 构建（此前未设置，等于无优化编译）
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -O3：最大化优化；-march=native：针对部署机/机器人固定 CPU 生成指令（二者均为原生编译场景）
add_compile_options(-O3 -march=native)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# OpenVINO 路径：优先命令行传入，其次部署机 /opt/intel，最后系统安装
if(NOT OpenVINO_DIR)
  if(EXISTS "/opt/intel/openvino_2024.6.0/runtime/cmake")
    set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
  elseif(EXISTS "/usr/lib/cmake/openvino2024.6.0")
    set(OpenVINO_DIR "/usr/lib/cmake/openvino2024.6.0")
  endif()
endif()
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# TensorRT（WSL 本机 NVIDIA 显卡推理）：头文件 / 库 + CUDA runtime
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include /opt/TensorRT/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib /opt/TensorRT/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found. 请安装 TensorRT 或手动指定 TENSORRT_INCLUDE_DIR / TENSORRT_LIBRARY")
endif()
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h HINTS /usr/local/cuda/include)
find_library(CUDA_RUNTIME_LIBRARY cudart HINTS /usr/local/cuda/lib64)
if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_RUNTIME_LIBRARY)
  message(FATAL_ERROR "CUDA runtime not found. 请安装 CUDA 或手动指定 CUDA_INCLUDE_DIR / CUDA_RUNTIME_LIBRARY")
endif()

# 第三方库（自带 CMakeLists，内部不动）
add_subdirectory(third_party/serial)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})

# tinympc 依赖 Eigen 头文件路径，需在 include_directories 之后加入
add_subdirectory(third_party/tinympc)

# 相机厂商 SDK 头文件
include_directories(third_party/camera_sdk/hikrobot/include)
include_directories(third_party/camera_sdk/mindvision/include)

#################### 图像预处理库 ####################
# STATIC：作为基础工具模块被 auto_aim/auto_buff 等 OBJECT 库复用，
# 需要 .a 以便依赖沿 target_link_libraries 的 PUBLIC 链自动传递
add_library(image STATIC
    src/image/image.cpp
)

target_link_libraries(image PUBLIC ${OpenCV_LIBS})

#################### 灯条处理库 ####################
# STATIC：同上，被 auto_aim 的 detector/classifier 复用
add_library(lightbar STATIC
    src/lightbar/lightbar.cpp
)

target_link_libraries(lightbar PUBLIC ${OpenCV_LIBS} image)

#################### 工具库 ####################
add_library(tools OBJECT
    src/tools/exiter.cpp
    src/tools/extended_kalman_filter.cpp
    src/tools/ransac_sine_fitter.cpp
    src/tools/img_tools.cpp
    src/tools/math_tools.cpp
    src/tools/plotter.cpp
    src/tools/trajectory.cpp
    src/tools/recorder.cpp
    src/tools/logger.cpp
    src/tools/pid.cpp
    src/tools/crc.cpp
)

#################### IO 库（相机 + 通信） ####################
add_library(io OBJECT
    # 相机模块
    src/camera/camera.cpp
    src/camera/usbcamera.cpp
    src/camera/video_camera.cpp
    src/camera/hikrobot.cpp
    src/camera/mindvision.cpp
    # 通信模块
    src/comm/gimbal.cpp
    src/comm/cboard.cpp
    src/comm/dm_imu.cpp
)

# 相机厂商 SDK 库目录与链接
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(SDK_ARCH amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SDK_ARCH arm64)
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}!")
endif()
link_directories(third_party/camera_sdk/hikrobot/lib/${SDK_ARCH})
link_directories(third_party/camera_sdk/mindvision/lib/${SDK_ARCH})

target_link_libraries(io PUBLIC MvCameraControl MVSDK usb-1.0 yaml-cpp serial)

#################### 自瞄任务库 ####################
add_library(auto_aim OBJECT
    # 装甲板数据模型
    src/armor/armor.cpp
    # 数字分类器
    src/classifier/classifier.cpp
    # 检测器（传统 + YOLO）
    src/detector/detector.cpp
    src/detector/yolo.cpp
    src/detector/tensorrt_infer.cpp
    src/detector/yolos/yolov5.cpp
    src/detector/yolos/yolov8.cpp
    src/detector/yolos/yolo11.cpp
    src/detector/multithread/commandgener.cpp
    src/detector/multithread/mt_detector.cpp
    # 解算模块
    src/solve/solver.cpp
    # 跟踪模块（EKF 目标模型）
    src/tracker/target.cpp
    src/tracker/tracker.cpp
    # 开火模块（瞄准 + 开火判定）
    src/fire/aimer.cpp
    src/fire/shooter.cpp
    # MPC 规划器
    src/planner/planner.cpp
)

target_link_libraries(auto_aim PUBLIC io openvino::runtime tinympcstatic lightbar image
    ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})

#################### 全向感知任务库 ####################
add_library(omni OBJECT
    src/omni/perceptron.cpp
    src/omni/decider.cpp
)

target_link_libraries(omni PUBLIC io auto_aim)

#################### 能量机关任务库 ####################
add_library(auto_buff OBJECT
    src/buff/buff_type.cpp
    src/buff/buff_detector.cpp
    src/buff/buff_solver.cpp
    src/buff/buff_target.cpp
    src/buff/buff_aimer.cpp
    src/buff/yolo11_buff.cpp
)

target_link_libraries(auto_buff PUBLIC io auto_aim image)

#################### 主程序入口 ####################
add_executable(standard apps/standard.cpp)
add_executable(mt_standard apps/mt_standard.cpp)
add_executable(standard_mpc apps/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc apps/auto_aim_debug_mpc.cpp)
add_executable(mt_auto_aim_debug apps/mt_auto_aim_debug.cpp)
add_executable(traditional_debug apps/traditional_debug.cpp)

foreach(app standard mt_standard standard_mpc auto_aim_debug_mpc mt_auto_aim_debug traditional_debug)
    target_link_libraries(${app} ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
endforeach()

# 精简传统流水线（无硬件视频调试入口）
add_executable(MAIN apps/main.cpp)
target_link_libraries(MAIN ${OpenCV_LIBS} yaml-cpp serial fmt::fmt spdlog::spdlog
    tools io auto_aim omni)

#################### 冒烟测试（临时，验证用） ####################
add_executable(yolo_smoke smoke/yolo_smoke.cpp)
target_link_libraries(yolo_smoke ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

#################### 标定工具 ####################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt spdlog::spdlog tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt spdlog::spdlog yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt spdlog::spdlog yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt spdlog::spdlog yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt spdlog::spdlog tools)

#################### 测试 ####################
set(TEST_BASE_LIBS ${OpenCV_LIBS} fmt::fmt spdlog::spdlog yaml-cpp tools io)

add_executable(auto_aim_test tests/auto_aim_test.cpp)
add_executable(auto_buff_test tests/auto_buff_test.cpp)
add_executable(camera_detect_test tests/camera_detect_test.cpp)
add_executable(camera_test tests/camera_test.cpp)
add_executable(camera_thread_test tests/camera_thread_test.cpp)
add_executable(cboard_test tests/cboard_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(multi_usbcamera_test tests/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(minimum_vision_system tests/minimum_vision_system.cpp)
add_executable(camera_auto_aim_test tests/camera_auto_aim_test.cpp)
add_executable(gimbal_test tests/gimbal_test.cpp)
add_executable(planner_test tests/planner_test.cpp)
add_executable(planner_test_offline tests/planner_test_offline.cpp)

foreach(t   auto_aim_test
            camera_detect_test
            camera_thread_test
            detector_video_test
            usbcamera_detect_test
            handeye_test
            minimum_vision_system
            camera_auto_aim_test
            gimbal_test
            planner_test
            planner_test_offline)
    target_link_libraries(${t} ${TEST_BASE_LIBS} auto_aim)

endforeach()
target_link_libraries(auto_buff_test ${TEST_BASE_LIBS} auto_aim auto_buff)

foreach(t   camera_test
            cboard_test
            fire_test
            gimbal_response_test
            multi_usbcamera_test
            usbcamera_test dm_test)
    target_link_libraries(${t} ${TEST_BASE_LIBS})

endforeach()

# ROS2 相关测试（需要 ROS 环境，默认不编译）
# add_executable(publish_test tests/publish_test.cpp)
# add_executable(subscribe_test tests/subscribe_test.cpp)
# add_executable(topic_loop_test tests/topic_loop_test.cpp)
