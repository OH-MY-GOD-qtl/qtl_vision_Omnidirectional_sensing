// traditional_debug：传统视觉检测 + MPC 规划 + 串口云台，带检测调试窗口
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "fire/aimer.hpp"
#include "detector/detector.hpp"  // 使用传统检测器
#include "detector/multithread/commandgener.hpp"
#include "planner/planner.hpp"
#include "fire/shooter.hpp"
#include "solve/solver.hpp"
#include "tracker/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || !cli.has("@config-path")) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;
    Recorder recorder;

    Gimbal gimbal(config_path);
    Camera camera(config_path);

    Detector detector(config_path, true);  // 使用传统检测器替代YOLO
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Planner planner(config_path);

    ThreadSafeQueue<std::optional<Target>, true> target_queue(1);
    target_queue.push(std::nullopt);

    std::atomic<bool> quit = false;

    std::atomic<GimbalMode> mode{GimbalMode::IDLE};
    auto last_mode{GimbalMode::IDLE};

    auto plan_thread = std::thread([&]() {
        auto t0 = std::chrono::steady_clock::now();
        auto last_fps_time = std::chrono::steady_clock::now();
        int frame_count = 0;

        while (!quit) {
            if (!target_queue.empty() && mode == GimbalMode::AUTO_AIM) {
                auto target = target_queue.front();
                auto gs = gimbal.state();
                auto plan = planner.plan(target, gs.bullet_speed);
                auto is_empty = target.has_value() ? false : true;

                gimbal.send(
                    plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
                    plan.pitch_acc, is_empty);
                nlohmann::json data;

                // 计算帧率
                frame_count++;
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time).count();
                double fps = 0.0;
                if (elapsed >= 1000) {
                    fps = frame_count * 1000.0 / elapsed;
                    frame_count = 0;
                    last_fps_time = current_time;
                }

                data["t"] = delta_time(std::chrono::steady_clock::now(), t0);
                data["control"] = plan.control;
                data["fire"] = plan.fire;
                data["plan_yaw"] = plan.yaw * 57.3;
                data["gimbal_yaw"] = gs.yaw * 57.3;
                data["plan_pitch"] = plan.pitch * 57.3;
                data["gimbal_pitch"] = gs.pitch * 57.3;
                data["is_empty"] = is_empty;
                data["bullet_speed"] = gs.bullet_speed;
                data["fps"] = fps;  // 在data中添加帧率信息

                std::this_thread::sleep_for(10ms);
            } else
                std::this_thread::sleep_for(200ms);
        }
    });

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
        mode = gimbal.mode();

        if (last_mode != mode) {
            logger()->info("Switch to {}", gimbal.str(mode));
            last_mode = mode.load();
        }

        camera.read(img, t);
        auto q = gimbal.q(t);
        solver.set_R_gimbal2world(q);

        //recorder.record(img, q, t);

        /// 自瞄
        if (mode.load() == GimbalMode::AUTO_AIM) {
            auto armors = detector.detect(img);  // 使用传统检测方法

            // 绘制装甲板
            for (const auto &armor : armors) {
                if (armor.points.size() != 4) continue;
                // 绘制装甲板边界框
                for (int i = 0; i < 4; i++) {
                    cv::line(img, 
                                        cv::Point(armor.points[i].x, armor.points[i].y), 
                                        cv::Point(armor.points[(i+1)%4].x, armor.points[(i+1)%4].y), 
                                        {0, 255, 0}, 2);
                }
            }

            auto targets = tracker.track(armors, t, gimbal.state().id);
            if (!targets.empty())
                target_queue.push(targets.front());
            else
                target_queue.push(std::nullopt);
        }

        else
            gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

        // 显示检测窗口
        cv::resize(img, img, {}, 0.5, 0.5);  // 缩小图片尺寸以便显示
        cv::imshow("Detection", img);
        auto key = cv::waitKey(1);
        if (key == 'q') break;
    }

    quit = true;
    if (plan_thread.joinable()) plan_thread.join();
    gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    cv::destroyAllWindows();  // 关闭窗口

    return 0;
}