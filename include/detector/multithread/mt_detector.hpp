#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>

#include "detector/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"


namespace multithread
{

class MultiThreadDetector
{
public:
    MultiThreadDetector(const std::string & config_path, bool debug = false);

    void push(cv::Mat img, std::chrono::steady_clock::time_point t);

    std::tuple<std::vector<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

    std::tuple<cv::Mat, std::vector<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
    // 推理请求池大小：即最大 in-flight 帧数，兼顾吞吐与内存占用
    static constexpr int kPoolSize = 4;

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    std::string device_;
    YOLO yolo_;

    // 复用池：避免每帧 create_infer_request() 与 640x640 输入缓冲分配
    std::vector<ov::InferRequest> infer_requests_;
    std::vector<cv::Mat> inputs_;
    // 空闲槽位队列：push 时取槽位，pop 完成后归还
    ThreadSafeQueue<int> free_slots_{kPoolSize};

    // 结果队列仅存槽位索引 + 原始帧引用 + 时间戳；容量 = 池大小，
    // push 前已持有空闲槽位时入队必成功，不会覆盖 in-flight 请求
    ThreadSafeQueue<std::tuple<int, cv::Mat, std::chrono::steady_clock::time_point>>
        queue_{kPoolSize, [] { logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread


#endif  // AUTO_AIM__MT_DETECTOR_HPP