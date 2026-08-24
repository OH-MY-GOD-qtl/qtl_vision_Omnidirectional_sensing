#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "camera/camera.hpp"
#include "fire/aimer.hpp"
#include "solve/solver.hpp"
#include "tracker/tracker.hpp"
#include "detector/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys = 
    "{help h usage ? |                              | 输出命令行参数说明 }" 
    "{@config-path   | configs/camera_auto_aim.yaml | 相机测试自瞄配置文件的路径}" ;

int main(int argc, char * argv[])
{
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    auto camera_auto_aim_test_config_path = cli.get<std::string>(0);

    Plotter plotter;
    Exiter exiter;

    // 初始化相机
    Camera camera(camera_auto_aim_test_config_path);

    // 初始化自瞄组件
    YOLO yolo(camera_auto_aim_test_config_path);
    Solver solver(camera_auto_aim_test_config_path);
    Tracker tracker(camera_auto_aim_test_config_path, solver);
    Aimer aimer(camera_auto_aim_test_config_path);

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    int frame_count = 0;

    Target last_target;
    double last_t = -1;

    while (!exiter.exit()) {
        // 从相机读取图像
        camera.read(img, timestamp);
        if (img.empty()) {
            std::cerr << "相机读取失败" << std::endl;
            continue;
        }

        frame_count++;

        /// 自瞄核心逻辑

        auto yolo_start = std::chrono::steady_clock::now();
        auto armors = yolo.detect(img, frame_count);

        auto tracker_start = std::chrono::steady_clock::now();
        auto targets = tracker.track(armors, timestamp);

        auto aimer_start = std::chrono::steady_clock::now();
        // 调用aimer.aim获取瞄准点，但不处理命令输出
        aimer.aim(targets, timestamp, 27, false);

        /// 调试输出

        auto finish = std::chrono::steady_clock::now();
        logger()->info(
            "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
            delta_time(tracker_start, yolo_start) * 1e3,
            delta_time(aimer_start, tracker_start) * 1e3,
            delta_time(finish, aimer_start) * 1e3);

        // 绘制装甲板检测结果
        for (const auto& armor : armors) {
            cv::rectangle(img, armor.box, {255, 0, 0}, 2);// 绘制装甲板框
            draw_text(img, fmt::format("{}", ARMOR_NAMES[armor.name]), armor.box.tl(), {255, 255, 255});
        }

        nlohmann::json data;

        // 装甲板原始观测数据
        data["armor_num"] = armors.size();
        if (!armors.empty()) {
            const auto & armor = armors.front();
            data["armor_x"] = armor.xyz_in_world[0];
            data["armor_y"] = armor.xyz_in_world[1];
            data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
            data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
            data["armor_center_x"] = armor.center_norm.x;
            data["armor_center_y"] = armor.center_norm.y;
        }

        if (!targets.empty()) {
            auto target = targets.front();

            if (last_t == -1) {
                last_target = target;
                last_t = 0;
                continue;
            }

            std::vector<Eigen::Vector4d> armor_xyza_list;

            // 当前帧target更新后
            armor_xyza_list = target.armor_xyza_list();
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
            if (aim_point.valid) draw_points(img, image_points, {0, 0, 255});

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
        }

        plotter.plot(data);


            cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
            cv::imshow("cream_auto_aim", img);
            if (cv::waitKey(1) == 'q') break;

    }

    return 0;
}