#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "comm/ros2/ros2.hpp"
#include "armor/armor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

int main(int argc, char ** argv)
{
    Exiter exiter;
    ROS2 ros2;

    double i = 0;
    while (!exiter.exit()) {
        Eigen::Vector4d data{i, i + 1, 1, ArmorName::sentry + 1};
        ros2.publish(data);
        i++;

        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (i > 1000) break;
    }
    return 0;
}
