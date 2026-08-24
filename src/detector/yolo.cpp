#include "detector/yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "detector/yolos/yolo11.hpp"
#include "detector/yolos/yolov5.hpp"
#include "detector/yolos/yolov8.hpp"
#include "tools/yaml.hpp"


YOLO::YOLO(const std::string & config_path, bool debug)
{
    auto yaml = yaml_load(config_path);
    auto yolo_name = yaml_read<std::string>(yaml, "yolo_name");

    if (yolo_name == "yolov8") {
        yolo_ = std::make_unique<YOLOV8>(config_path, debug);
    }

    else if (yolo_name == "yolo11") {
        yolo_ = std::make_unique<YOLO11>(config_path, debug);
    }

    else if (yolo_name == "yolov5") {
        yolo_ = std::make_unique<YOLOV5>(config_path, debug);
    }

    else {
        throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
    }
}

std::vector<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
    return yolo_->detect(img, frame_count);
}

std::vector<Armor> YOLO::postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
    return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

