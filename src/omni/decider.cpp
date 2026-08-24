#include "omni/decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/yaml.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"


Decider::Decider(const std::string & config_path) : detector_(config_path), count_(0)
{
    auto yaml = yaml_load(config_path);
    img_width_ = yaml_read<double>(yaml, "image_width");
    img_height_ = yaml_read<double>(yaml, "image_height");
    fov_h_ = yaml_read<double>(yaml, "fov_h");
    fov_v_ = yaml_read<double>(yaml, "fov_v");
    new_fov_h_ = yaml_read<double>(yaml, "new_fov_h");
    new_fov_v_ = yaml_read<double>(yaml, "new_fov_v");
    enemy_color_ =
        (yaml_read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue;
    mode_ = yaml_read<double>(yaml, "mode");
}

Command Decider::decide(
    YOLO & yolo, const Eigen::Vector3d & gimbal_pos, USBCamera & usbcam1,
    USBCamera & usbcam2, Camera & back_camera)
{
    Eigen::Vector2d delta_angle;
    USBCamera * cams[] = {&usbcam1, &usbcam2};

    cv::Mat usb_img;
    std::chrono::steady_clock::time_point timestamp;
    if (count_ < 0 || count_ > 2) {
        throw std::runtime_error("count_ out of valid range [0,2]");
    }
    if (count_ == 2) {
        back_camera.read(usb_img, timestamp);
    } else {
        cams[count_]->read(usb_img, timestamp);
    }
    auto armors = yolo.detect(usb_img);
    auto empty = armor_filter(armors);

    if (!empty) {
        if (count_ == 2) {
            delta_angle = this->delta_angle(armors, "back");
        } else {
            delta_angle = this->delta_angle(armors, cams[count_]->device_name);
        }

        logger()->debug(
            "[{} camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
            (count_ == 2 ? "back" : cams[count_]->device_name), delta_angle[0], delta_angle[1],
            armors.size(), ARMOR_NAMES[armors.front().name]);

        count_ = (count_ + 1) % 3;

        return Command{
            true, false, limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
            limit_rad(delta_angle[1] / 57.3)};
    }

    count_ = (count_ + 1) % 3;
    // 如果没有找到目标，返回默认命令
    return Command{false, false, 0, 0};
}

Command Decider::decide(
    YOLO & yolo, const Eigen::Vector3d & gimbal_pos, Camera & back_cammera)
{
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    back_cammera.read(img, timestamp);
    auto armors = yolo.detect(img);
    auto empty = armor_filter(armors);

    if (!empty) {
        auto delta_angle = this->delta_angle(armors, "back");
        logger()->debug(
            "[back camera] delta yaw:{:.2f},target pitch:{:.2f},armor number:{},armor name:{}",
            delta_angle[0], delta_angle[1], armors.size(), ARMOR_NAMES[armors.front().name]);

        return Command{
            true, false, limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
            limit_rad(delta_angle[1] / 57.3)};
    }

    return Command{false, false, 0, 0};
}

Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
    if (detection_queue.empty()) {
        return Command{false, false, 0, 0};
    }

    DetectionResult dr = detection_queue.front();
    if (dr.armors.empty()) return Command{false, false, 0, 0};
    logger()->info(
        "omniperceptron find {},delta yaw is {:.4f}", ARMOR_NAMES[dr.armors.front().name],
        dr.delta_yaw * 57.3);

    return Command{true, false, dr.delta_yaw, dr.delta_pitch};
};

Eigen::Vector2d Decider::delta_angle(
    const std::vector<Armor> & armors, const std::string & camera)
{
    Eigen::Vector2d delta_angle;
    if (camera == "left") {
        delta_angle[0] = 62 + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
        delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
        return delta_angle;
    }

    else if (camera == "right") {
        delta_angle[0] = -62 + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
        delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
        return delta_angle;
    }

    else {
        delta_angle[0] = 170 + (54.2 / 2) - armors.front().center_norm.x * 54.2;
        delta_angle[1] = armors.front().center_norm.y * 44.5 - 44.5 / 2;
        return delta_angle;
    }
}

bool Decider::armor_filter(std::vector<Armor> & armors)
{
    if (armors.empty()) return true;
    // 过滤非敌方装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor & a) { return a.color != enemy_color_; }),
        armors.end());

    // 25赛季没有5号装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor & a) { return a.name == ArmorName::five; }),
        armors.end());
    // 不打工程
    // armors.erase(std::remove_if(armors.begin(), armors.end(),
    //   [&](const Armor & a) { return a.name == ArmorName::two; }), armors.end());
    // 不打前哨站
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor & a) { return a.name == ArmorName::outpost; }),
        armors.end());

    // 过滤掉刚复活无敌的装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor & a) {
                return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
                                invincible_armor_.end();
            }),
        armors.end());

    return armors.empty();
}

void Decider::set_priority(std::vector<Armor> & armors)
{
    if (armors.empty()) return;

    const PriorityMap & priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;

    if (!armors.empty()) {
        for (auto & armor : armors) {
            armor.priority = priority_map.at(armor.name);
        }
    }
}

void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
    if (detection_queue.empty()) return;

    // 对每个 DetectionResult 调用 armor_filter 和 set_priority
    for (auto & dr : detection_queue) {
        armor_filter(dr.armors);
        set_priority(dr.armors);

        // 对每个 DetectionResult 中的 armors 进行排序
        std::sort(
            dr.armors.begin(), dr.armors.end(),
            [](const Armor & a, const Armor & b) { return a.priority < b.priority; });
    }

    // 根据优先级对 DetectionResult 进行排序
    std::sort(
        detection_queue.begin(), detection_queue.end(),
        [](const DetectionResult & a, const DetectionResult & b) {
            return a.armors.front().priority < b.armors.front().priority;
        });
}

Eigen::Vector4d Decider::get_target_info(
    const std::vector<Armor> & armors, const std::vector<Target> & targets)
{
    if (armors.empty() || targets.empty()) return Eigen::Vector4d::Zero();

    auto target = targets.front();

    for (const auto & armor : armors) {
        if (armor.name == target.name) {
            return Eigen::Vector4d{
                armor.xyz_in_gimbal[0], armor.xyz_in_gimbal[1], 1,
                static_cast<double>(armor.name) + 1};  //避免歧义+1(详见通信协议)
        }
    }

    return Eigen::Vector4d::Zero();
}

void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
    invincible_armor_.clear();

    if (invincible_enemy_ids.empty()) return;

    for (const auto & id : invincible_enemy_ids) {
        logger()->info("invincible armor id: {}", id);
        invincible_armor_.push_back(ArmorName(id - 1));
    }
}

void Decider::get_auto_aim_target(
    std::vector<Armor> & armors, const std::vector<int8_t> & auto_aim_target)
{
    if (auto_aim_target.empty()) return;

    std::vector<ArmorName> auto_aim_targets;

    for (const auto & target : auto_aim_target) {
        if (target <= 0 || static_cast<size_t>(target) > ARMOR_NAMES.size()) {
            logger()->warn("Received invalid auto_aim target value: {}", int(target));
            continue;
        }
        auto_aim_targets.push_back(static_cast<ArmorName>(target - 1));
        logger()->info("nav send auto_aim target is {}", ARMOR_NAMES[target - 1]);
    }

    if (auto_aim_targets.empty()) return;

    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor & a) {
                return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
                                auto_aim_targets.end();
            }),
        armors.end());
}

