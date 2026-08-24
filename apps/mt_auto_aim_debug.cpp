// mt_auto_aim_debug：多线程检测自瞄（YOLO 多线程 + 弹道瞄准 + 串口云台），带重投影调试窗口
#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera.hpp"
#include "comm/gimbal.hpp"  // 替换cboard.hpp为gimbal.hpp
#include "fire/aimer.hpp"
#include "detector/multithread/commandgener.hpp"
#include "detector/multithread/mt_detector.hpp"
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

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | | 位置参数，yaml配置文件路径 }";

using namespace std::chrono;

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
    Recorder recorder(100);  //根据实际帧率调整

    Gimbal gimbal(config_path);  // 替换cboard为gimbal
    Camera camera(config_path);

    multithread::MultiThreadDetector detector(config_path, true);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Aimer aimer(config_path);
    Shooter shooter(config_path);
    multithread::CommandGener commandgener(shooter, aimer, gimbal, plotter, true);  // 创建CommandGener实例，开启debug模式

    auto detect_thread = std::thread([&]() {
        cv::Mat img;
        std::chrono::steady_clock::time_point t;

        while (!exiter.exit()) {
            camera.read(img, t);
            detector.push(img, t);
        }
    });

    auto mode = Mode::idle;
    auto last_mode = Mode::idle;

    while (!exiter.exit()) {
        auto t0 = std::chrono::steady_clock::now();
        /// 自瞄核心逻辑
        auto [img, armors, t] = detector.debug_pop();
        Eigen::Quaterniond q = gimbal.q(t-1ms);  // 取上一帧的数据,补偿传输延时
        mode = static_cast<Mode>(static_cast<int>(gimbal.mode()));  // 转换gimbal模式到io::Mode

        if (last_mode != mode) {
            logger()->info("Switch to {}", gimbal.str(gimbal.mode()));  // 使用gimbal的str方法获取模式字符串
            last_mode = mode;
        }

        solver.set_R_gimbal2world(q);

        Eigen::Vector3d ypr = eulers(solver.R_gimbal2world(), 2, 1, 0);

        auto targets = tracker.track(armors, t);

        // 通过commandgener发送命令
        commandgener.push(targets, t, gimbal.state().bullet_speed, ypr);



        nlohmann::json data;
        data["t"] = delta_time(std::chrono::steady_clock::now(), t0);

            /// debug
        // 绘制自瞄状态文本
        draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});
        //绘制GimabalToVision数据
        draw_text(
            img,
            fmt::format(
                "Gimbal Mode: {}, Yaw: {:.2f}, Pitch: {:.2f}, Bullet Speed: {:.2f}",
                gimbal.str(gimbal.mode()), ypr[0] * 57.3, ypr[1] * 57.3,
                gimbal.state().bullet_speed),
            {10, 60}, {255, 255, 0});

        //draw_point()
        // 绘制当前帧率
        //auto fps = 1.0 / delta_time(std::chrono::steady_clock::now(), t0);
        // draw_text(img, fmt::format("FPS: {:.2f}", fps), {10, 120}, {255, 255, 255});
        //控制台打印帧数
        //logger()->info("Current FPS: {:.2f}", fps);
                //绘制画面中心点
        cv::circle(img, cv::Point(img.cols / 2, img.rows / 2), 5, {255, 0, 0}, -1);

        // 连接识别到装甲板的对角线并比较中心
        for (const auto &armor : armors) {
            if( armor.points.size() !=4) continue;
            // 连接对角线
            cv::line(img, cv::Point(armor.points[0].x, armor.points[0].y), 
            cv::Point(armor.points[2].x, armor.points[2].y), {0, 255, 0}, 2);
            cv::line(img, cv::Point(armor.points[1].x, armor.points[1].y), 
            cv::Point(armor.points[3].x, armor.points[3].y), {0, 255, 0}, 2);

        }
        // 装甲板原始观测数据
        data["armor_num"] = armors.size();
        if (!armors.empty()) {
            auto min_x = 1e10;
            auto & armor = armors.front();
            for (auto & a : armors) {
                if (a.center.x < min_x) {
                    min_x = a.center.x;
                    armor = a;
                }
            }  //always left
            solver.solve(armor);
            data["armor_x"] = armor.xyz_in_world[0];
            data["armor_y"] = armor.xyz_in_world[1];
            data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
            data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
        }

        if (!targets.empty()) {
            auto target = targets.front();

            // 当前帧target更新后
            std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
            for (const Eigen::Vector4d & xyza : armor_xyza_list) {
                auto image_points =
                    solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
                draw_points(img, image_points, {0, 255, 0});
            }

            // aimer瞄准位置
            auto aim_point = aimer.debug_aim_point;
            Eigen::Vector4d aim_xyza = aim_point.xyza;
            auto image_points =
                solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
            if (aim_point.valid)
                draw_points(img, image_points, {0, 0, 255});
            else
                draw_points(img, image_points, {255, 0, 0});

            // 观测器内部数据
            Eigen::VectorXd x = target.ekf_x();
            data["x"] = x[0];
            data["vx"] = x[1];
            data["y"] = x[2];
            data["vy"] = x[3];
            data["z"] = x[4];
            data["vz"] = x[5];
            data["a"] = x[6] * 57.3;
            data["w"] = x[7];
            data["r"] = x[8];
            data["l"] = x[9];
            data["h"] = x[10];
            data["last_id"] = target.last_id;

            // 卡方检验数据
            data["residual_yaw"] = target.ekf().data.at("residual_yaw");
            data["residual_pitch"] = target.ekf().data.at("residual_pitch");
            data["residual_distance"] = target.ekf().data.at("residual_distance");
            data["residual_angle"] = target.ekf().data.at("residual_angle");
            data["nis"] = target.ekf().data.at("nis");
            data["nees"] = target.ekf().data.at("nees");
            data["nis_fail"] = target.ekf().data.at("nis_fail");
            data["nees_fail"] = target.ekf().data.at("nees_fail");
            data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
        }

        // 云台响应情况
        data["gimbal_yaw"] = ypr[0] * 57.3;
        data["gimbal_pitch"] = ypr[1] * 57.3;
        //data["bullet_speed"] = gimbal.state().bullet_speed;  // 使用gimbal的state获取子弹速度


        plotter.plot(data);

        cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
        cv::imshow("reprojection", img);
        auto key = cv::waitKey(1);
        if (key == 'q') break;
    }

    detect_thread.join();

    return 0;
}