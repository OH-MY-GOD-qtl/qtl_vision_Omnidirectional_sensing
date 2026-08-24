#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <optional>

#include "buff/buff_type.hpp"
#include "tools/img_tools.hpp"
#include "buff/yolo11_buff.hpp"
const int LOSE_MAX = 20;  // 丢失的阙值

class Buff_Detector
{
public:
    Buff_Detector(const std::string & config);

    std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

    std::optional<PowerRune> detect(cv::Mat & bgr_img);

std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
    void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

    cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

    void handle_lose();

    YOLO11_BUFF MODE_;
    Track_status status_;
    int lose_;  // 丢失的次数
    double lastlen_;
    std::optional<PowerRune> last_powerrune_ = std::nullopt;

    // 预处理中间缓冲：帧间复用，避免重复分配
    cv::Mat gray_, binary_;
};
#endif  // DETECTOR_HPP