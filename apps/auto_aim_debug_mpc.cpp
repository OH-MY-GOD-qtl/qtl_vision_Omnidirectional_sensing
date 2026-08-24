// auto_aim_debug_mpc：YOLO + EKF 跟踪 + MPC 规划 + 串口云台，带重投影调试窗口
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "comm/gimbal.hpp"
#include "planner/planner.hpp"
#include "solve/solver.hpp"
#include "tracker/tracker.hpp"
#include "detector/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/standard1.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
    Exiter exiter;
    Plotter plotter;

    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    Gimbal gimbal(config_path);
    Camera camera(config_path);

    YOLO yolo(config_path, true);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Planner planner(config_path);

    ThreadSafeQueue<std::optional<Target>, true> target_queue(1);
    target_queue.push(std::nullopt);

    std::atomic<bool> quit = false;
    auto plan_thread = std::thread([&]() {
        auto t0 = std::chrono::steady_clock::now();

        while (!quit) {
            auto target = target_queue.front();
            auto gs = gimbal.state();
            auto plan = planner.plan(target, gs.bullet_speed);
            auto is_empty = target.has_value() ? false : true;

            gimbal.send(
                plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
                plan.pitch_acc, is_empty);

            nlohmann::json data;
            data["t"] = delta_time(std::chrono::steady_clock::now(), t0);

            data["gimbal_yaw"] = gs.yaw * 57.2958;  // 世界坐标系中云台当前yaw角度（单位：度）
            //data["gimbal_yaw_vel"] = gs.yaw_vel;
            data["gimbal_pitch"] = gs.pitch * 57.2958;  // 世界坐标系中云台当前pitch角度（单位：度）
            //data["gimbal_pitch_vel"] = gs.pitch_vel;

            data["target_yaw"] = plan.target_yaw * 57.2958;  //目标在世界坐标系中的yaw角度（单位：度）
            data["target_pitch"] =
                plan.target_pitch * 57.2958;  //目标在世界坐标系中的pitch角度（单位：度）

            data["plan_yaw"] = plan.yaw * 57.2958;  // 期望在世界坐标系中的yaw角度（单位：度）
            // data["plan_yaw_vel"] = plan.yaw_vel;
            // data["plan_yaw_acc"] = plan.yaw_acc;

            data["plan_pitch"] = plan.pitch * 57.2958;  // 期望在世界坐标系中的pitch角度（单位：度）
            // data["plan_pitch_vel"] = plan.pitch_vel;
            // data["plan_pitch_acc"] = plan.pitch_acc;

            data["fire"] = plan.fire ? 1 : 0;
            data["is_empty"] = is_empty ? 1 : 0;
            data["bullet_speed"] = gs.bullet_speed;
            data["robot_id"] = gs.id;
            data["mode"] = plan.control ? (plan.fire ? 2 : 1) : 0;
            if (target.has_value()) {
                data["target_z"] = target->ekf_x()[4];   //z目标在世界坐标系中的z轴位置坐标
                data["target_vz"] = target->ekf_x()[5];  //vz目标在z轴方向的速度分量
            }

            if (target.has_value()) {
                data["w"] = target->ekf_x()[7];
            } else {
                data["w"] = 0.0;
            }

            float diff_yaw = (plan.yaw - gs.yaw)*57.2958;
            float diff_pitch = (plan.pitch - gs.pitch)*57.2958;
            data["diff_yaw"] = diff_yaw;
            data["diff_pitch"] = diff_pitch;
            data["distance"] = target.has_value() ? std::hypot(target->ekf_x()[0], target->ekf_x()[2]) : 0.0;

            plotter.plot(data);

            std::this_thread::sleep_for(10ms);
        }
    });

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
        camera.read(img, t);
        auto q = gimbal.q(t);

        solver.set_R_gimbal2world(q);
        auto armors = yolo.detect(img);
        auto targets = tracker.track(armors, t,gimbal.state().id);
        if (!targets.empty())
            target_queue.push(targets.front());
        else
            target_queue.push(std::nullopt);

        if (!targets.empty()) {//绘制目标投影
            auto target = targets.front();

            // 当前帧target更新后
            std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
            for (const Eigen::Vector4d & xyza : armor_xyza_list) {
                auto image_points =
                    solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
                draw_points(img, image_points, {0, 255, 0});
            }

            Eigen::Vector4d aim_xyza = planner.debug_xyza;
            auto image_points =
                solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
            draw_points(img, image_points, {0, 0, 255});
        }

        cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
        cv::imshow("reprojection", img);
        auto key = cv::waitKey(1);
        if (key == 'q') break;
    }

    quit = true;
    if (plan_thread.joinable()) plan_thread.join();
    gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);

    return 0;
}