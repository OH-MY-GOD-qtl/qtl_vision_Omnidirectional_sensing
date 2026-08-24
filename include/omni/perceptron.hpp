#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <chrono>
#include <vector>
#include <memory>

#include "omni/decider.hpp"
#include "omni/detection.hpp"
#include "camera/usbcamera.hpp"
#include "armor/armor.hpp"
#include "tools/thread_safe_queue.hpp"



class Perceptron
{
public:
    Perceptron(
        USBCamera * usbcma1, USBCamera * usbcam2, USBCamera * usbcam3,
        USBCamera * usbcam4, const std::string & config_path);

    ~Perceptron();

    std::vector<DetectionResult> get_detection_queue();

    void parallel_infer(USBCamera * cam, std::shared_ptr<YOLO> & yolo_parallel);

private:
    std::vector<std::thread> threads_;
    ThreadSafeQueue<DetectionResult> detection_queue_;

    std::shared_ptr<YOLO> yolo_parallel1_;
    std::shared_ptr<YOLO> yolo_parallel2_;
    std::shared_ptr<YOLO> yolo_parallel3_;
    std::shared_ptr<YOLO> yolo_parallel4_;

    Decider decider_;
    bool stop_flag_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

#endif