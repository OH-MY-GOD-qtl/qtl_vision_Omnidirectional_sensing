#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/usbcamera.hpp"
#include "detector/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
    "{@name          | video0                 | 端口名称 }"
    "{d display      |                        | 显示视频流       }";

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    Exiter exiter;

    auto config_path = cli.get<std::string>(0);
    auto device_name = cli.get<std::string>(1);
    auto display = cli.has("display");

    USBCamera usbcam(device_name, config_path);

    YOLO yolo(config_path, true);

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    auto last_stamp = std::chrono::steady_clock::now();
    while (!exiter.exit()) {
        usbcam.read(img, timestamp);
        yolo.detect(img);

        auto dt = delta_time(timestamp, last_stamp);
        last_stamp = timestamp;

        logger()->info("{:.2f} fps", 1 / dt);
        std::this_thread::sleep_for(10ms);

        if (!display) continue;
        cv::imshow("img", img);
        if (cv::waitKey(1) == 'q') break;
    }
}