#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <vector>
#include <optional>

#include "tracker/target.hpp"
#include "tinympc/tiny_api.hpp"


constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using StateTraj = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
    bool control;
    bool fire;
    float target_yaw;
    float target_pitch;
    float yaw;
    float yaw_vel;
    float yaw_acc;
    float pitch;
    float pitch_vel;
    float pitch_acc;
};

class Planner
{
public:
    Eigen::Vector4d debug_xyza;
    Planner(const std::string & config_path);

    Plan plan(Target & target, double bullet_speed);
    Plan plan(std::optional<Target> & target, double bullet_speed);

// 在Planner类的private部分添加基础偏移量成员变量
private:
    // 距离相关的偏移量配置，格式：distance -> (yaw_offset, pitch_offset)
    std::vector<std::tuple<double, double, double>> distance_offsets_;
    // 基础偏移量配置（当没有距离相关配置时使用）
    double yaw_offset_;
    double pitch_offset_;

    double fire_thresh_;
    double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;

    TinySolver * yaw_solver_;
    TinySolver * pitch_solver_;

    void setup_yaw_solver(const std::string & config_path);
    void setup_pitch_solver(const std::string & config_path);

    // 根据距离计算插值后的偏移量
    std::pair<double, double> get_offset_by_distance(double distance);

    Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
    StateTraj get_trajectory(Target & target, double yaw0, double bullet_speed);
};


#endif  // AUTO_AIM__PLANNER_HPP