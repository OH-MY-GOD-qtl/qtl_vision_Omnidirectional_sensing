#include "fire/shooter.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/yaml.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"


Shooter::Shooter(const std::string & config_path) : last_command_{false, false, 0, 0}
{
    auto yaml = yaml_load(config_path);
    first_tolerance_ = yaml_read<double>(yaml, "first_tolerance") / 57.3;    // degree to rad
    second_tolerance_ = yaml_read<double>(yaml, "second_tolerance") / 57.3;  // degree to rad
    judge_distance_ = yaml_read<double>(yaml, "judge_distance");
    auto_fire_ = yaml_read<bool>(yaml, "auto_fire");
}

bool Shooter::shoot(
    const Command & command, const Aimer & aimer,
    const std::vector<Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
    if (!command.control || targets.empty() || !auto_fire_) return false;

    auto target_x = targets.front().ekf_x()[0];
    auto target_y = targets.front().ekf_x()[2];
    auto tolerance = std::sqrt(square(target_x) + square(target_y)) > judge_distance_
                                            ? second_tolerance_
                                            : first_tolerance_;
    // logger()->debug("d(command.yaw) is {:.4f}", std::abs(last_command_.yaw - command.yaw));
    if (
        std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&  //此时认为command突变不应该射击
        std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&    //应该减去上一次command的yaw值
        aimer.debug_aim_point.valid) {
        last_command_ = command;
        return true;
    }

    last_command_ = command;
    return false;
}

