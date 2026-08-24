#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <string>
#include <vector>

#include "armor/armor.hpp"


// 检测结果来源：USB(传统检测，仅报有无目标) / HIK(海康模型，主瞄准)
enum class DetectionSource
{
    USB,
    HIK
};

//一个识别结果可能包含多个armor,需要排序和过滤。armors, timestamp, delta_yaw, delta_pitch
struct DetectionResult
{
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    double delta_yaw;    //rad
    double delta_pitch;  //rad
    DetectionSource source = DetectionSource::USB;  // 结果来源
    std::string camera;                            // 相机标识: usb0/usb1/usb2/hik

    // Assignment operator
    DetectionResult & operator=(const DetectionResult & other)
    {
        if (this != &other) {
            armors = other.armors;
            timestamp = other.timestamp;
            delta_yaw = other.delta_yaw;
            delta_pitch = other.delta_pitch;
            source = other.source;
            camera = other.camera;
        }
        return *this;
    }
};

#endif