#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <iostream>
#include <vector>
#include <unordered_map>

#include "omni/detection.hpp"
#include "camera/camera.hpp"
#include "comm/command.hpp"
#include "camera/usbcamera.hpp"
#include "armor/armor.hpp"
#include "tracker/target.hpp"
#include "detector/yolo.hpp"


class Decider
{
public:
    Decider(const std::string & config_path);

    Command decide(
        YOLO & yolo, const Eigen::Vector3d & gimbal_pos, USBCamera & usbcam1,
        USBCamera & usbcam2, Camera & back_cammera);

    Command decide(
        YOLO & yolo, const Eigen::Vector3d & gimbal_pos, Camera & back_cammera);

    Command decide(const std::vector<DetectionResult> & detection_queue);

    Eigen::Vector2d delta_angle(
        const std::vector<Armor> & armors, const std::string & camera);

    bool armor_filter(std::vector<Armor> & armors);

    void set_priority(std::vector<Armor> & armors);
    //对队列中的每一个DetectionResult进行过滤，同时将DetectionResult排序
    void sort(std::vector<DetectionResult> & detection_queue);

    Eigen::Vector4d get_target_info(
        const std::vector<Armor> & armors, const std::vector<Target> & targets);

    void get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids);

    void get_auto_aim_target(
        std::vector<Armor> & armors, const std::vector<int8_t> & auto_aim_target);

private:
    int img_width_;
    int img_height_;
    double fov_h_, new_fov_h_;
    double fov_v_, new_fov_v_;
    int mode_;
    int count_;

    // 全向感知相机安装参数
    std::vector<double> usb_yaw_offsets_;  // 三个USB相对yaw零位的安装角(degree)
    double hik_yaw_offset_;                // 海康相对yaw零位安装角(degree)
    double hik_fov_h_;                     // 海康水平FOV(degree)
    double hik_fov_v_;                     // 海康垂直FOV(degree)
    double pitch_preset_;                  // 无海康目标时的pitch预设(degree)

    Color enemy_color_;
    YOLO detector_;
    std::vector<ArmorName> invincible_armor_;  //无敌状态机器人编号,英雄为1，哨兵为6

    // 定义ArmorName到ArmorPriority的映射类型
    using PriorityMap = std::unordered_map<ArmorName, ArmorPriority>;

    const PriorityMap mode1 = {
        {ArmorName::one, ArmorPriority::second},
        {ArmorName::two, ArmorPriority::forth},
        {ArmorName::three, ArmorPriority::first},
        {ArmorName::four, ArmorPriority::first},
        {ArmorName::five, ArmorPriority::third},
        {ArmorName::sentry, ArmorPriority::third},
        {ArmorName::outpost, ArmorPriority::fifth},
        {ArmorName::base, ArmorPriority::fifth},
        {ArmorName::not_armor, ArmorPriority::fifth}};

    const PriorityMap mode2 = {
        {ArmorName::two, ArmorPriority::first},
        {ArmorName::one, ArmorPriority::second},
        {ArmorName::three, ArmorPriority::second},
        {ArmorName::four, ArmorPriority::second},
        {ArmorName::five, ArmorPriority::second},
        {ArmorName::sentry, ArmorPriority::third},
        {ArmorName::outpost, ArmorPriority::third},
        {ArmorName::base, ArmorPriority::third},
        {ArmorName::not_armor, ArmorPriority::third}};
};

enum PriorityMode
{
    MODE_ONE = 1,
    MODE_TWO
};


#endif