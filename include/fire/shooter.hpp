#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>

#include "comm/command.hpp"
#include "fire/aimer.hpp"


class Shooter
{
public:
    Shooter(const std::string & config_path);

    bool shoot(
        const Command & command, const Aimer & aimer,
        const std::vector<Target> & targets, const Eigen::Vector3d & gimbal_pos);

private:
    Command last_command_;
    double judge_distance_;
    double first_tolerance_;
    double second_tolerance_;
    bool auto_fire_;
};

#endif  // AUTO_AIM__SHOOTER_HPP