#ifndef VIDEO_CAMERA_HPP
#define VIDEO_CAMERA_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "camera/camera.hpp"

// 视频文件相机：无硬件时用录制视频跑通全链路（读到结尾循环播放）
class VideoCamera : public CameraBase {
public:
        explicit VideoCamera(const std::string& config_path);

        void read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp) override;

private:
        cv::VideoCapture cap_;
        std::string video_path_;
};

#endif
