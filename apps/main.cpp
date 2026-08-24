// MAIN：精简传统流水线（传统检测 + 弹道瞄准 + 串口云台），带调试窗口；默认 configs/standard1.yaml
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <string>

#include "camera/camera.hpp"
#include "comm/gimbal.hpp"
#include "detector/detector.hpp"
#include "fire/aimer.hpp"
#include "fire/shooter.hpp"
#include "solve/solver.hpp"
#include "tools/logger.hpp"
#include "tracker/tracker.hpp"

int main(int argc, char** argv)
{
    // 配置文件路径：命令行参数或默认 configs/standard1.yaml
    std::string config_path = (argc > 1) ? argv[1] : "configs/standard1.yaml";

    Camera camera(config_path);
    Gimbal gimbal(config_path);
    Detector detector(config_path, true);  // 传统检测器 + 调试窗口
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Aimer aimer(config_path);
    Shooter shooter(config_path);

    logger()->info("[Main] qtl_vision 启动，配置文件: {}", config_path);

    cv::Mat img;
    while (true) {
        std::chrono::steady_clock::time_point t;
        camera.read(img, t);
        if (img.empty()) continue;

        auto gs = gimbal.state();
        auto mode = gimbal.mode();

        if (mode == GimbalMode::AUTO_AIM) {
            // 识别流水线：检测 → 位姿解算 → 跟踪 → 瞄准 → 开火判定
            solver.set_R_gimbal2world(gimbal.q(t));

            auto armors = detector.detect(img);
            for (auto & armor : armors) solver.solve(armor);

            auto targets = tracker.track(armors, t, gs.id);
            auto command = aimer.aim(targets, t, gs.bullet_speed, true);
            bool fire = shooter.shoot(command, aimer, targets, Eigen::Vector3d(gs.yaw, 0, gs.pitch));

            // 节流状态日志（无显示环境也能验证链路）
            static int log_count = 0;
            if (++log_count % 30 == 0)
                logger()->info(
                    "[Main] state={} armors={} targets={} fire={}", tracker.state(), armors.size(),
                    targets.size(), fire);

            gimbal.send(command.control, fire, command.yaw, 0, 0, command.pitch, 0, 0,
                                    targets.empty());
        }
        else {
            gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);
        }

        if (cv::waitKey(1) == 'q') break;
    }

    gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);
    logger()->info("[Main] 退出");
    return 0;
}
