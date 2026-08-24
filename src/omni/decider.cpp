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

    // 全向感知相机安装参数（缺失时使用默认值，兼容旧配置）
    usb_yaw_offsets_ = {60, -60, -180};
    if (yaml["omni_usb_yaw_offsets"]) {
        usb_yaw_offsets_.clear();
        for (const auto & v : yaml["omni_usb_yaw_offsets"]) {
            usb_yaw_offsets_.push_back(v.as<double>());
        }
    }
    hik_yaw_offset_ = yaml["omni_hik_yaw_offset"] ? yaml["omni_hik_yaw_offset"].as<double>() : 0.0;
    hik_fov_h_ = yaml["omni_hik_fov_h"] ? yaml["omni_hik_fov_h"].as<double>() : 54.2;
    hik_fov_v_ = yaml["omni_hik_fov_v"] ? yaml["omni_hik_fov_v"].as<double>() : 44.5;
    pitch_preset_ = yaml["omni_pitch_preset"] ? yaml["omni_pitch_preset"].as<double>() : 0.0;
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

    // 对结果过滤 + 优先级排序（不改变外部传入队列）
    auto queue = detection_queue;
    sort(queue);

    // 海康优先：海康有目标 -> yaw/pitch 全由海康驱动
    for (const auto & dr : queue) {
        if (dr.source == DetectionSource::HIK) {
            if (dr.armors.empty()) continue;
            logger()->info(
                "omni[HIK] find {},delta yaw is {:.4f},delta pitch is {:.4f}",
                ARMOR_NAMES[dr.armors.front().name], dr.delta_yaw * 57.3, dr.delta_pitch * 57.3);
            return Command{true, false, dr.delta_yaw, dr.delta_pitch};
        }
    }

    // 无海康目标：用 USB 引导 yaw，pitch 固定为预设值
    for (const auto & dr : queue) {
        if (dr.source == DetectionSource::USB) {
            if (dr.armors.empty()) continue;
            logger()->info(
                "omni[USB:{}] find target,delta yaw is {:.4f},pitch preset",
                dr.camera, dr.delta_yaw * 57.3);
            return Command{true, false, dr.delta_yaw, pitch_preset_ / 57.3};
        }
    }

    return Command{false, false, 0, 0};
}

Eigen::Vector2d Decider::delta_angle(
    const std::vector<Armor> & armors, const std::string & camera)
{
    // 相机安装 yaw 偏移 + 各自 FOV（degree）
    double yaw_offset = 0.0;
    double fov_h = fov_h_;
    double fov_v = fov_v_;

    if (camera == "hik") {
        yaw_offset = hik_yaw_offset_;
        fov_h = hik_fov_h_;
        fov_v = hik_fov_v_;
    } else if (camera.rfind("usb", 0) == 0 && camera.size() > 3) {
        int idx = std::stoi(camera.substr(3));
        if (idx >= 0 && idx < static_cast<int>(usb_yaw_offsets_.size())) {
            yaw_offset = usb_yaw_offsets_[idx];
        }
    }

    Eigen::Vector2d delta_angle;
    delta_angle[0] = yaw_offset + (fov_h / 2.0) - armors.front().center_norm.x * fov_h;
    delta_angle[1] = armors.front().center_norm.y * fov_v - fov_v / 2.0;
    return delta_angle;
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

