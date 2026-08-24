#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "planner/planner.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

using namespace std::chrono_literals;

const std::string keys =
    "{help h usage ? |     | 输出命令行参数说明    }"
    "{@config-path   |     | yaml配置文件路径     }"
    "{@d             | 3.0 | Target距离(m)        }"
    "{@w             | 5.0 | Target角速度(rad/s)  }";

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    auto d = cli.get<double>(1);
    auto w = cli.get<double>(2);
    if (cli.has("help") || !cli.has("@config-path")) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;
    Plotter plotter;

    Planner planner(config_path);
    Target target(d, w, 0.2, 0.1);

    auto t0 = std::chrono::steady_clock::now();

    while (!exiter.exit()) {
        target.predict(0.01);

        auto plan = planner.plan(target, 22);

        nlohmann::json data;
        data["t"] = delta_time(std::chrono::steady_clock::now(), t0);

        data["target_yaw"] = plan.target_yaw;
        data["target_pitch"] = plan.target_pitch;

        data["plan_yaw"] = plan.yaw;
        data["plan_yaw_vel"] = plan.yaw_vel;
        data["plan_yaw_acc"] = plan.yaw_acc;

        data["plan_pitch"] = plan.pitch;
        data["plan_pitch_vel"] = plan.pitch_vel;
        data["plan_pitch_acc"] = plan.pitch_acc;

        plotter.plot(data);

        std::this_thread::sleep_for(10ms);
    }

    return 0;
}