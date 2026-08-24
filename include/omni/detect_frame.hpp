#ifndef OMNIPERCEPTION__DETECT_FRAME_HPP
#define OMNIPERCEPTION__DETECT_FRAME_HPP

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "detector/yolo.hpp"
#include "tools/logger.hpp"


// 多相机检测帧：与业务(检测结果)强耦合，故从 tools 下移至此
struct Frame
{
    int id;
    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    Eigen::Quaterniond q;
    std::vector<Armor> armors;
};

inline std::vector<YOLO> create_yolo11s(
    const std::string & config_path, int numebr, bool debug)
{
    std::vector<YOLO> yolo11s;
    for (int i = 0; i < numebr; i++) {
        yolo11s.push_back(YOLO(config_path, debug));
    }
    return yolo11s;
}

inline std::vector<YOLO> create_yolov8s(
    const std::string & config_path, int numebr, bool debug)
{
    std::vector<YOLO> yolov8s;
    for (int i = 0; i < numebr; i++) {
        yolov8s.push_back(YOLO(config_path, debug));
    }
    return yolov8s;
}

// 按帧 id 顺序输出的有序队列
class OrderedQueue
{
public:
    OrderedQueue() : current_id_(1) {}
    ~OrderedQueue()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            main_queue_ = std::queue<Frame>();
            buffer_.clear();
            current_id_ = 0;
        }
        logger()->info("OrderedQueue destroyed, queue and buffer cleared.");
    }

    void enqueue(const Frame & item)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (item.id < current_id_) {
            logger()->warn("small id");
            return;
        }

        if (item.id == current_id_) {
            main_queue_.push(item);
            current_id_++;

            auto it = buffer_.find(current_id_);
            while (it != buffer_.end()) {
                main_queue_.push(it->second);
                buffer_.erase(it);
                current_id_++;
                it = buffer_.find(current_id_);
            }

            if (main_queue_.size() >= 1) {
                cond_var_.notify_one();
            }
        } else {
            buffer_[item.id] = item;
        }
    }

    Frame dequeue()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        cond_var_.wait(lock, [this]() { return !main_queue_.empty(); });

        Frame item = main_queue_.front();
        main_queue_.pop();
        return item;
    }

    // 不会阻塞队列
    bool try_dequeue(Frame & item)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (main_queue_.empty()) {
            return false;
        }
        item = main_queue_.front();
        main_queue_.pop();
        return true;
    }

    size_t get_size() { return main_queue_.size() + buffer_.size(); }

private:
    std::queue<Frame> main_queue_;
    std::unordered_map<int, Frame> buffer_;
    int current_id_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};


#endif  // OMNIPERCEPTION__DETECT_FRAME_HPP
