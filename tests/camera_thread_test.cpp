#include <fmt/core.h>
#include <unistd.h>

#include <chrono>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "comm/cboard.hpp"
#include "fire/aimer.hpp"
#include "solve/solver.hpp"
#include "tracker/tracker.hpp"
#include "detector/yolo.hpp"
#include "omni/detect_frame.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_pool.hpp"

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/ascento.yaml | 位置参数，yaml配置文件路径 }";

OrderedQueue frame_queue;

// 处理detect任务的线程函数
void detect_frame(Frame && frame, YOLO & yolo)
{
    frame.armors = yolo.detect(frame.img);
    frame_queue.enqueue(frame);
}

int main(int argc, char * argv[])
{
    Exiter exiter;
    Plotter plotter;
    // Recorder recorder(100);

    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    // 处理线程函数
    auto process_thread = std::thread([&]() {
        Frame process_frame;
        while (!exiter.exit()) {
            process_frame = frame_queue.dequeue();
            auto img = process_frame.img;
            auto armors = process_frame.armors;
            auto t = process_frame.t;

            nlohmann::json data;
            data["armor_num"] = armors.size();

            plotter.plot(data);
            // cv::resize(img, img, {}, 0.5, 0.5);
            // cv::imshow("reprojection", img);
        }
    });

    Camera camera(config_path);
    int num_yolo_thread = 8;
    auto yolos = create_yolov8s(config_path, num_yolo_thread, true);
    // auto yolos = create_yolo11s(config_path, num_yolo_thread, true);
    std::vector<bool> yolo_used(num_yolo_thread, false);
    ThreadPool thread_pool(num_yolo_thread);

    cv::Mat img;
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point t;
    std::chrono::steady_clock::time_point last_t = std::chrono::steady_clock::now();

    int frame_id = 0;

    while (!exiter.exit()) {
        camera.read(img, t);
        auto dt = delta_time(t, last_t);
        last_t = t;

        // logger()->info("{:.2f} fps", 1 / dt);
        // draw_text(img, fmt::format("{:.2f} fps", 1/dt), {10, 60}, {255, 255, 255});
        nlohmann::json data;
        data["fps"] = 1 / dt;

        frame_id++;

        // 将处理任务提交到线程池
        std::mutex yolo_mutex;
        thread_pool.enqueue([&, frame_id, t] {
            YOLO * yolo = nullptr;
            int yolo_id = -1;
            for (int i = 0; i < num_yolo_thread; i++) {
                if (!yolo_used[i]) {
                    yolo_used[i] = true;
                    yolo = &yolos[i];
                    yolo_id = i;
                    break;
                }
            }
            if (yolo) {
                Frame frame{frame_id, img.clone(), t};
                detect_frame(std::move(frame), *yolo);

                yolo_used[yolo_id] = false;
            }
        });
        plotter.plot(data);

        auto key = cv::waitKey(1);
        if (key == 'q') break;
    }

    return 0;
}
