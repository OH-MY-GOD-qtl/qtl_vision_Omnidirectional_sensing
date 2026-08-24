#include "omni/perceptron.hpp"

#include <chrono>
#include <memory>
#include <thread>

#include "detector/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"


Perceptron::Perceptron(
    USBCamera * usbcam1, USBCamera * usbcam2, USBCamera * usbcam3,
    Camera * hikcam, const std::string & config_path)
: detection_queue_(10), decider_(config_path), stop_flag_(false)
{
    // 海康一路 YOLO
    yolo_ = std::make_shared<YOLO>(config_path, false);

    // USB 三路传统检测：每个线程独立 Detector 实例（Detector 含帧级缓冲，非线程安全）
    detector1_ = std::make_shared<Detector>(config_path, false, false);  // 不加载分类器
    detector2_ = std::make_shared<Detector>(config_path, false, false);
    detector3_ = std::make_shared<Detector>(config_path, false, false);

    // USB 安装角: +60 / -60 / -180，对应 usb0/usb1/usb2
    threads_.emplace_back([this, usbcam1] { parallel_infer_usb(usbcam1, *detector1_, "usb0"); });
    threads_.emplace_back([this, usbcam2] { parallel_infer_usb(usbcam2, *detector2_, "usb1"); });
    threads_.emplace_back([this, usbcam3] { parallel_infer_usb(usbcam3, *detector3_, "usb2"); });
    threads_.emplace_back([this, hikcam] { parallel_infer_hik(hikcam, *yolo_); });

    logger()->info("Perceptron initialized (3 USB traditional + 1 Hik YOLO).");
}

Perceptron::~Perceptron()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_flag_ = true;  // 设置退出标志
    }
    condition_.notify_all();  // 唤醒所有等待的线程

    // 等待线程结束
    for (auto & t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
    std::vector<DetectionResult> result;
    DetectionResult temp;

    // 注意：这里的 pop 不阻塞（假设队列为空时会报错或忽略）
    while (!detection_queue_.empty()) {
        detection_queue_.pop(temp);
        result.push_back(std::move(temp));
    }

    return result;
}

// USB 传统检测线程
void Perceptron::parallel_infer_usb(
    USBCamera * cam, Detector & detector, const std::string & camera_key)
{
    if (!cam) {
        logger()->error("USB camera pointer is null!");
        return;
    }
    try {
        while (true) {
            cv::Mat img;
            std::chrono::steady_clock::time_point ts;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (stop_flag_) break;  // 检查是否需要退出
            }

            cam->read(img, ts);
            if (img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            auto armors = detector.detect(img);
            if (!armors.empty()) {
                auto delta_angle = decider_.delta_angle(armors, camera_key);

                DetectionResult dr;
                dr.armors = std::move(armors);
                dr.timestamp = ts;
                dr.delta_yaw = delta_angle[0] / 57.3;
                dr.delta_pitch = delta_angle[1] / 57.3;
                dr.source = DetectionSource::USB;
                dr.camera = camera_key;
                detection_queue_.push(dr);  // 推入线程安全队列
            }
        }
    } catch (const std::exception & e) {
        logger()->error("Exception in parallel_infer_usb: {}", e.what());
    }
}

// 海康模型检测线程
void Perceptron::parallel_infer_hik(Camera * cam, YOLO & yolo)
{
    if (!cam) {
        logger()->error("Hik camera pointer is null!");
        return;
    }
    try {
        while (true) {
            cv::Mat img;
            std::chrono::steady_clock::time_point ts;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (stop_flag_) break;  // 检查是否需要退出
            }

            cam->read(img, ts);
            if (img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            auto armors = yolo.detect(img);
            if (!armors.empty()) {
                auto delta_angle = decider_.delta_angle(armors, "hik");

                DetectionResult dr;
                dr.armors = std::move(armors);
                dr.timestamp = ts;
                dr.delta_yaw = delta_angle[0] / 57.3;
                dr.delta_pitch = delta_angle[1] / 57.3;
                dr.source = DetectionSource::HIK;
                dr.camera = "hik";
                detection_queue_.push(dr);  // 推入线程安全队列
            }
        }
    } catch (const std::exception & e) {
        logger()->error("Exception in parallel_infer_hik: {}", e.what());
    }
}

