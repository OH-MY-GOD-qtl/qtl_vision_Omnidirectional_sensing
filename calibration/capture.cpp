#include <fmt/core.h>

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "camera/camera.hpp"
#include "comm/gimbal.hpp"
#include "tools/yaml.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "yaml-cpp/yaml.h"

const std::string keys =
    "{help h usage ?  |                          | 输出命令行参数说明}"
    "{@config-path    | configs/calibration.yaml | yaml配置文件路径 }"
    "{@output-folder  | assets/img_with_q        | 输出文件夹路径   }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
    std::ofstream q_file(q_path);
    Eigen::Vector4d xyzw = q.coeffs();
    // 输出顺序为wxyz
    q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
    q_file.close();
}

void capture_loop(int pattern_cols, int pattern_rows,
    const std::string & config_path, const std::string & output_folder)
{
    Gimbal gimbal(config_path);
    Camera camera(config_path);
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;

    int count = 0;
    while (true) {
        camera.read(img, timestamp);
        Eigen::Quaterniond q = gimbal.q(timestamp);

        // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
        auto img_with_ypr = img.clone();
        Eigen::Vector3d zyx = eulers(q, 2, 1, 0) * 57.3;  // degree
        draw_text(img_with_ypr, fmt::format("Yaw {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
        draw_text(img_with_ypr, fmt::format("Pitch {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
        draw_text(img_with_ypr, fmt::format("Roll {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

        // draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
        // draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
        // draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

        std::vector<cv::Point2f> centers_2d;
        //auto success = cv::findCirclesGrid(img, cv::Size(pattern_cols, pattern_rows), centers_2d);  // 默认是对称圆点图案
        //cv::drawChessboardCorners(img_with_ypr, cv::Size(pattern_cols, pattern_rows), centers_2d, success);  // 显示识别结果

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // 检测方形棋盘格角点
        auto success = cv::findChessboardCorners(gray, cv::Size(11, 8), centers_2d, 
                                                                                cv::CALIB_CB_ADAPTIVE_THRESH | 
                                                                                cv::CALIB_CB_NORMALIZE_IMAGE |
                                                                                cv::CALIB_CB_FAST_CHECK);

        // 如果检测成功，可以进一步精确角点位置
        if (success) {
                cv::cornerSubPix(gray, centers_2d, cv::Size(11, 11), cv::Size(-1, -1), 
                                                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        }

        // 绘制检测结果
        cv::drawChessboardCorners(img_with_ypr, cv::Size(11, 8), centers_2d, success);
        cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

        // 按“s”保存图片和对应四元数，按“q”退出程序
        cv::imshow("Press s to save, q to quit", img_with_ypr);
        auto key = cv::waitKey(1);
        if (key == 'q')
            break;
        else if (key != 's')
            continue;

        // 保存图片和四元数
        count++;
        auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
        auto q_path = fmt::format("{}/{}.txt", output_folder, count);
        cv::imwrite(img_path, img);
        write_q(q_path, q);
        logger()->info("[{}] Saved in {}", count, output_folder);
    }

    // 离开该作用域时，camera和gimbal会自动关闭
}

int main(int argc, char * argv[])
{
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    auto config_path = cli.get<std::string>(0);
    auto output_folder = cli.get<std::string>(1);
    // 新建输出文件夹
    std::filesystem::create_directory(output_folder);
    auto yaml = yaml_load(config_path);
    auto pattern_cols = yaml["pattern_cols"];
    auto pattern_rows = yaml["pattern_rows"];
    logger()->info("标定板尺寸为列行{}x{},中心间距{}mm", pattern_cols.as<int>(), pattern_rows.as<int>(), yaml_read<double>(yaml, "center_distance_mm"));
    // 主循环，保存图片和对应四元数 


    capture_loop(pattern_cols.as<int>()-1, pattern_rows.as<int>()-1, config_path, output_folder);

    logger()->warn("注意四元数输出顺序为wxyz");

    return 0;
}