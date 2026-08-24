// standard_mpc：YOLO 自瞄 + MPC 规划 + 串口云台，无显示窗口
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "fire/aimer.hpp"
#include "detector/multithread/commandgener.hpp"
#include "detector/multithread/mt_detector.hpp"
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

    YOLO yolo(config_path, true);
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
        uint16_t last_bullet_count = 0;

        while (!quit) {
            if (!target_queue.empty() && mode == GimbalMode::AUTO_AIM) {
                auto target = target_queue.front();
                auto gs = gimbal.state();
                auto plan = planner.plan(target, gs.bullet_speed);
                auto is_empty = target.has_value() ? false : true;

                gimbal.send(
                    plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
                    plan.pitch_acc, is_empty);

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
        recorder.record(img, q, t);
        solver.set_R_gimbal2world(q);

        /// 自瞄
        if (mode.load() == GimbalMode::AUTO_AIM) {
            auto armors = yolo.detect(img);
            auto targets = tracker.track(armors, t,gimbal.state().id);
            if (!targets.empty())
                target_queue.push(targets.front());
            else
                target_queue.push(std::nullopt);
        }

        else
            gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    }

    quit = true;
    if (plan_thread.joinable()) plan_thread.join();
    gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

    return 0;
}