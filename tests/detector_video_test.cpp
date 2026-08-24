#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "detector/detector.hpp"
#include "detector/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明 }"
    "{@video_path    |                        | avi路径}"
    "{@config-path   | configs/sentry.yaml    | yaml配置文件的路径}"
    "{@start-index   | 0                      | 视频起始帧下标    }"
    "{@end-index     | 0                      | 视频结束帧下标    }"
    "{tradition t    |  false                 | 是否使用传统方法识别}";

int main(int argc, char * argv[])
{
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    auto video_path = cli.get<std::string>(0);
    auto config_path = cli.get<std::string>(1);
    auto start_index = cli.get<int>(2);
    auto end_index = cli.get<int>(3);
    auto use_tradition = cli.get<bool>("tradition");

    Exiter exiter;
    Plotter plotter;

    cv::VideoCapture video(video_path);

    Detector detector(config_path);
    YOLO yolo(config_path);

    video.set(cv::CAP_PROP_POS_FRAMES, start_index);

    for (int frame_count = start_index; !exiter.exit(); frame_count++) {
        if (end_index > 0 && frame_count > end_index) break;

        cv::Mat img;
        std::vector<Armor> armors;
        video.read(img);
        if (img.empty()) break;
        // cv::GaussianBlur(img, img, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);

        if (use_tradition)
            armors = detector.detect(img, frame_count);
        else
            armors = yolo.detect(img, frame_count);

        // 在图像上显示帧数
        std::string frame_text = fmt::format("Frame: {}", frame_count);
        cv::putText(img, frame_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // 显示检测结果
        for (const auto& armor : armors) {
            for (int i = 0; i < 4; i++) {
                cv::line(img, armor.points[i], armor.points[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
            }
        }

        // 显示图像
        cv::imshow("Detection", img);

        if (!armors.empty()) {
            nlohmann::json data;
            auto armor = armors.front();

            data["armor_0_pixel_x"] = armor.points[0].x;
            data["armor_0_pixel_y"] = armor.points[0].y;
            data["armor_1_pixel_x"] = armor.points[1].x;
            data["armor_1_pixel_y"] = armor.points[1].y;
            data["armor_2_pixel_x"] = armor.points[2].x;
            data["armor_2_pixel_y"] = armor.points[2].y;
            data["armor_3_pixel_x"] = armor.points[3].x;
            data["armor_3_pixel_y"] = armor.points[3].y;
            plotter.plot(data);
        }

        auto key = cv::waitKey(33);
        if (key == 'q') break;
    }

    return 0;
}