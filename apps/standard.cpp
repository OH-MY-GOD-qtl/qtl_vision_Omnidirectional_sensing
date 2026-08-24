// standard：YOLO 自瞄 + 弹道瞄准，读取裁判系统 CBoard（CAN），无显示窗口
#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera.hpp"
#include "comm/cboard.hpp"
#include "fire/aimer.hpp"
#include "detector/multithread/commandgener.hpp"
#include "fire/shooter.hpp"
#include "solve/solver.hpp"
#include "tracker/tracker.hpp"
#include "detector/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明}"
    "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;
    Plotter plotter;
    Recorder recorder;

    CBoard cboard(config_path);
    Camera camera(config_path);

    YOLO detector(config_path, false);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Aimer aimer(config_path);
    Shooter shooter(config_path);

    cv::Mat img;
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point t;

    auto mode = Mode::idle;
    auto last_mode = Mode::idle;

    while (!exiter.exit()) {
        camera.read(img, t);
        q = cboard.imu_at(t - 1ms);
        mode = cboard.mode;

        if (last_mode != mode) {
            logger()->info("Switch to {}", MODES[mode]);
            last_mode = mode;
        }

        // recorder.record(img, q, t);

        solver.set_R_gimbal2world(q);

        Eigen::Vector3d ypr = eulers(solver.R_gimbal2world(), 2, 1, 0);

        auto armors = detector.detect(img);

        auto targets = tracker.track(armors, t);

        auto command = aimer.aim(targets, t, cboard.bullet_speed);

        cboard.send(command);
    }

    return 0;
}