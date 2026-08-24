#include <chrono>
#include <thread>

#include "comm/dm_imu.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

int main()
{
    Exiter exiter;
    DM_IMU imu;

    while (!exiter.exit()) {
        auto timestamp = std::chrono::steady_clock::now();

        std::this_thread::sleep_for(1ms);

        Eigen::Quaterniond q = imu.imu_at(timestamp);

        Eigen::Vector3d ypr = eulers(q, 2, 1, 0) * 57.3;
        logger()->info("z{:.2f} y{:.2f} x{:.2f} degree", ypr[0], ypr[1], ypr[2]);
    }

    return 0;
}