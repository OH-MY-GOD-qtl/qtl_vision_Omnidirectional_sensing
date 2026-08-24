#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <string>

#include "armor/armor.hpp"
#include "solve/solver.hpp"
#include "tracker/target.hpp"
#include "omni/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"


class Tracker
{
public:
    Tracker(const std::string & config_path, Solver & solver);

    std::string state() const;

    std::vector<Target> track(
        std::vector<Armor> & armors, std::chrono::steady_clock::time_point t, int id = 0,
        bool use_enemy_color = true);

    std::tuple<DetectionResult, std::vector<Target>> track(
        const std::vector<DetectionResult> & detection_queue, std::vector<Armor> & armors,
        std::chrono::steady_clock::time_point t, int id = 0, bool use_enemy_color = true);

    bool set_enemy_color(int id);
    Color get_enemy_color() const;

private:
    Solver & solver_;
    Color enemy_color_;
    int min_detect_count_;
    int max_temp_lost_count_;
    int detect_count_;
    int temp_lost_count_;
    int outpost_max_temp_lost_count_;
    int normal_temp_lost_count_;
    std::string state_, pre_state_;
    Target target_;
    std::chrono::steady_clock::time_point last_timestamp_;
    ArmorPriority omni_target_priority_;
    cv::Point2f img_center_;  // 图像中心坐标

    void state_machine(bool found);

    bool set_target(std::vector<Armor> & armors, std::chrono::steady_clock::time_point t);

    bool update_target(std::vector<Armor> & armors, std::chrono::steady_clock::time_point t);
};


#endif  // AUTO_AIM__TRACKER_HPP