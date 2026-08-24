// mt_standard：多线程检测自瞄（YOLO 多线程 + 弹道瞄准 + 串口云台），无显示窗口
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "comm/dm_imu.hpp"
#include "comm/gimbal.hpp"  // 添加Gimbal头文件
#include "fire/aimer.hpp"
#include "detector/multithread/commandgener.hpp"  // 添加CommandGener头文件
#include "detector/multithread/mt_detector.hpp"
#include "fire/shooter.hpp"
#include "solve/solver.hpp"
#include "tracker/tracker.hpp"
//#include "tasks/auto_buff/buff_aimer.hpp"
//#include "tasks/auto_buff/buff_detector.hpp"
//#include "tasks/auto_buff/buff_solver.hpp"
//#include "tasks/auto_buff/buff_target.hpp"
//#include "tasks/auto_buff/buff_type.hpp"
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

// 将GimbalMode转换为io::Mode
Mode to_io_mode(GimbalMode gimbal_mode) {
    switch (gimbal_mode) {
        case GimbalMode::AUTO_AIM:
            return Mode::auto_aim;
        case GimbalMode::SMALL_BUFF:
            return Mode::small_buff;
        case GimbalMode::BIG_BUFF:
            return Mode::big_buff;
        case GimbalMode::IDLE:
        default:
            return Mode::idle;
    }
}

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || !cli.has("@config-path")) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;
    Plotter plotter;
    Recorder recorder;

    Camera camera(config_path);
    Gimbal gimbal(config_path);  

    multithread::MultiThreadDetector detector(config_path);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Aimer aimer(config_path);
    Shooter shooter(config_path);
    multithread::CommandGener commandgener(shooter, aimer, gimbal, plotter,false);//debug开关
    //auto_buff::Buff_Detector buff_detector(config_path);
    //auto_buff::Solver buff_solver(config_path);
    //auto_buff::SmallTarget buff_small_target;
    //auto_buff::BigTarget buff_big_target;
    //auto_buff::Aimer buff_aimer(config_path);

    std::atomic<Mode> mode{Mode::idle};
    auto last_mode{Mode::idle};

    auto detect_thread = std::thread([&]() {
        cv::Mat img;
        std::chrono::steady_clock::time_point t;

        while (!exiter.exit()) {
            if (mode.load() == Mode::auto_aim) {
                camera.read(img, t);
                detector.push(img, t);
            } else {
                std::this_thread::sleep_for(10ms);
                continue;
            }
        }
    });

    while (!exiter.exit()) {
        // 从Gimbal获取当前模式
        auto gimbal_mode = gimbal.mode();
        mode = to_io_mode(gimbal_mode);

        if (last_mode != mode) {
            logger()->info("Switch to {}", MODES[mode]);
            last_mode = mode.load();
        }

        /// 自瞄
        if (mode.load() == Mode::auto_aim) {
            auto [img, armors, t] = detector.debug_pop();
            Eigen::Quaterniond q = gimbal.q(t - 1ms);  // 使用正确的q()方法获取四元数

            // recorder.record(img, q, t);

            solver.set_R_gimbal2world(q);

            Eigen::Vector3d ypr = eulers(solver.R_gimbal2world(), 2, 1, 0);

            auto targets = tracker.track(armors, t);

            commandgener.push(targets, t, gimbal.state().bullet_speed, ypr);  // 发送给决策线程
        }

        else {
            std::this_thread::sleep_for(10ms);
            continue;
        }
    }

    detect_thread.join();// 等待自瞄线程结束

    return 0;
}