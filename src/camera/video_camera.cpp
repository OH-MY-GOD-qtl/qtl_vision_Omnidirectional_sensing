#include "camera/video_camera.hpp"

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

VideoCamera::VideoCamera(const std::string& config_path) {
        auto yaml = yaml_load(config_path);
        video_path_ = yaml_read<std::string>(yaml, "video_path");

        if (!cap_.open(video_path_)) {
                logger()->error("[VideoCamera] 无法打开视频文件: {}", video_path_);
                exit(1);
        }
        logger()->info("[VideoCamera] 已打开视频文件: {}", video_path_);
}

void VideoCamera::read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp) {
        timestamp = std::chrono::steady_clock::now();
        cap_ >> img;
        if (img.empty()) {
                // 读到结尾则从头循环播放
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap_ >> img;
        }
}
