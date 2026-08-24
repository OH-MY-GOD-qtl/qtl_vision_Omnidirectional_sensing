#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "armor/armor.hpp"
#include "camera/camera.hpp"
#include "camera/usbcamera.hpp"
#include "detector/detector.hpp"
#include "detector/yolo.hpp"
#include "omni/decider.hpp"
#include "omni/detection.hpp"
#include "tools/thread_safe_queue.hpp"



// 全向感知：3 路 USB(传统检测，仅报有无目标) + 1 路海康(YOLO 模型，主瞄准)
// 4 相机共用 yaw；海康独享 pitch。
class Perceptron
{
public:
    Perceptron(
        USBCamera * usbcam1, USBCamera * usbcam2, USBCamera * usbcam3,
        Camera * hikcam, const std::string & config_path);

    ~Perceptron();

    std::vector<DetectionResult> get_detection_queue();

private:
    // USB 传统检测线程：cam -> Detector(灯条几何) -> DetectionResult(USB)
    void parallel_infer_usb(
        USBCamera * cam, Detector & detector, const std::string & camera_key);

    // 海康模型检测线程：cam -> YOLO -> DetectionResult(HIK)
    void parallel_infer_hik(Camera * cam, YOLO & yolo);

    std::vector<std::thread> threads_;
    ThreadSafeQueue<DetectionResult> detection_queue_;

    std::shared_ptr<YOLO> yolo_;            // 海康一路 YOLO
    std::shared_ptr<Detector> detector1_;   // USB 传统检测(每线程独立实例，Detector 非线程安全)
    std::shared_ptr<Detector> detector2_;
    std::shared_ptr<Detector> detector3_;

    Decider decider_;
    bool stop_flag_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

#endif