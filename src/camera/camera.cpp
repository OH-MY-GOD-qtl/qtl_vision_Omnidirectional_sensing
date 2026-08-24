#include "camera/camera.hpp"

#include <stdexcept>

#include "camera/hikrobot.hpp"
#include "camera/mindvision.hpp"
#include "camera/usbcamera.hpp"
#include "camera/video_camera.hpp"

#include "tools/yaml.hpp"


Camera::Camera(const std::string & config_path)
{
    auto yaml = yaml_load(config_path);
    auto camera_name = yaml_read<std::string>(yaml, "camera_name");
    auto exposure_ms = yaml_read<double>(yaml, "exposure_ms");

    if (camera_name == "mindvision") {
        auto gamma = yaml_read<double>(yaml, "gamma");
        auto vid_pid = yaml_read<std::string>(yaml, "vid_pid");
        camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
    }

    else if (camera_name == "hikrobot") {
        auto gain = yaml_read<double>(yaml, "gain");
        auto vid_pid = yaml_read<std::string>(yaml, "vid_pid");
        camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
    }

    else if (camera_name == "usb") {
        auto usb_device = yaml["usb_device"] ? yaml["usb_device"].as<std::string>() : "video0";
        camera_ = std::make_unique<USBCamera>(usb_device, config_path);
    }

    else if (camera_name == "video") {
        camera_ = std::make_unique<VideoCamera>(config_path);
    }

    else {
        throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
    }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
    camera_->read(img, timestamp);
}

