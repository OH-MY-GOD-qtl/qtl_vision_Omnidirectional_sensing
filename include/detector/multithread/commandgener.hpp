#ifndef AUTO_AIM_MULTITHREAD__HPP
#define AUTO_AIM_MULTITHREAD__HPP

#include <optional>

#include "comm/gimbal.hpp"
#include "fire/shooter.hpp"
#include "tracker/tracker.hpp"
#include "omni/decider.hpp"
#include "tools/plotter.hpp"


namespace multithread
{

class CommandGener
{
public:
    CommandGener(
        Shooter & shooter, Aimer & aimer, Gimbal & gimbal,
        Plotter & plotter, bool debug = false);

    ~CommandGener();

    void push(
        const std::vector<Target> & targets, const std::chrono::steady_clock::time_point & t,
        double bullet_speed, const Eigen::Vector3d & gimbal_pos);


private:
    struct Input
    {
        std::vector<Target> targets_;
        std::chrono::steady_clock::time_point t;
        // std::function<void()> decide;
        double bullet_speed;
        Eigen::Vector3d gimbal_pos;
    };

    Gimbal & gimbal_;
    Shooter & shooter_;
    Aimer & aimer_;
    Plotter & plotter_;

    std::optional<Input> latest_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread thread_;
    bool stop_, debug_;

    void generate_command();
};

}  // namespace multithread


#endif  // AUTO_AIM_MULTITHREAD__HPP