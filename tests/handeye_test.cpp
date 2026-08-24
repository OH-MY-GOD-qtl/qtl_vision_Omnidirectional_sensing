
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/core/eigen.hpp>

#include "camera/camera.hpp"
#include "comm/cboard.hpp"
#include "solve/solver.hpp"
#include "tools/yaml.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

const std::string keys =
    "{help h usage ? |                     | 输出命令行参数说明}"
    "{@config-path   | configs/handeye.yaml | yaml配置文件路径 }"
    "{d display      |                     | 显示视频流       }";

// 世界坐标到像素坐标的转换

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;

    auto config_path = cli.get<std::string>(0);
    auto display = cli.has("display");
    auto yaml = yaml_load(config_path);
    auto height = yaml_read<double>(yaml, "height");
    auto grid_num = yaml_read<int>(yaml, "grid_num");
    auto grid_size = yaml_read<double>(yaml, "grid_size");
    auto delay = yaml_read<int>(yaml, "delay");
    CBoard cboard(config_path);
    Camera camera(config_path);
    Solver solver(config_path);

    cv::Mat img;
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point t;
    std::vector<cv::Point3f> points;
    for (int x = 0; x < grid_num; x++) {
        for (int y = 0; y < grid_num; y++) {
            points.emplace_back(x * grid_size, y * grid_size - grid_num * grid_size / 2, -height);
            points.emplace_back(-x * grid_size, y * grid_size - grid_num * grid_size / 2, -height);
        }
    }
    while (!exiter.exit()) {
        camera.read(img, t);
        q = cboard.imu_at(t - 1ms * delay);
        solver.set_R_gimbal2world(q);
        cv::Mat result = img.clone();
        std::vector<cv::Point2f> projectedPoints = solver.world2pixel(points);
        for (const auto & point : projectedPoints) {
            if (point.x >= 0 && point.x < result.cols && point.y >= 0 && point.y < result.rows) {
                cv::circle(result, point, 3, cv::Scalar(255, 255, 255), -1);
            }
        }
        Eigen::Vector3d euler = solver.R_gimbal2world().eulerAngles(2, 1, 0) * 180.0 / M_PI;
        draw_text(result, fmt::format("yaw   {:.2f}", euler[0]), {40, 40}, {0, 0, 255});
        draw_text(result, fmt::format("pitch {:.2f}", euler[1]), {40, 80}, {0, 0, 255});
        draw_text(result, fmt::format("roll  {:.2f}", euler[2]), {40, 120}, {0, 0, 255});
        if (!display) continue;
        cv::imshow("result", result);
        if (cv::waitKey(1) == 'q') break;
    }
}