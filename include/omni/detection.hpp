#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <vector>

#include "armor/armor.hpp"


//一个识别结果可能包含多个armor,需要排序和过滤。armors, timestamp, delta_yaw, delta_pitch
struct DetectionResult
{
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    double delta_yaw;    //rad
    double delta_pitch;  //rad

    // Assignment operator
    DetectionResult & operator=(const DetectionResult & other)
    {
        if (this != &other) {
            armors = other.armors;
            timestamp = other.timestamp;
            delta_yaw = other.delta_yaw;
            delta_pitch = other.delta_pitch;
        }
        return *this;
    }
};

#endif