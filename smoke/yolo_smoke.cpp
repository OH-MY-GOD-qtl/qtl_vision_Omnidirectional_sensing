#include <opencv2/opencv.hpp>

#include <chrono>
#include <string>

#include "camera/camera.hpp"
#include "detector/yolo.hpp"
#include "tools/logger.hpp"

// 临时冒烟测试：验证 YOLO 模型加载 + 推理（绕开依赖 CAN 硬件的 cboard）
int main(int argc, char** argv)
{
    std::string config_path = (argc > 1) ? argv[1] : "configs/video_demo.yaml";

    logger()->info("=== YOLO 冒烟测试开始，配置: {} ===", config_path);

    Camera camera(config_path);

    logger()->info("正在加载 YOLO 模型（CPU 编译耗时较长）...");
    YOLO yolo(config_path, false);
    logger()->info("YOLO 模型加载完成");

    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    int frames = 0;
    int frames_with_armor = 0;

    while (frames < 300) {
        camera.read(img, t);
        if (img.empty()) continue;

        auto armors = yolo.detect(img, frames);
        if (!armors.empty()) {
            ++frames_with_armor;
            if (frames_with_armor % 10 == 1)
                logger()->info(
                    "frame {}: 检出 {} 块装甲板，第一块 name={} conf={:.2f}", frames, armors.size(),
                    ARMOR_NAMES[armors.front().name], armors.front().confidence);
        }
        ++frames;
    }

    logger()->info(
        "=== YOLO 冒烟测试完成：{} 帧中 {} 帧检出目标 ===", frames, frames_with_armor);
    return 0;
}
