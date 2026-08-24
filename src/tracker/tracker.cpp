#include "tracker/tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <tuple>

#include "tools/yaml.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"


Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
    detect_count_(0),
    temp_lost_count_(0),
    state_{"lost"},
    pre_state_{"lost"},
    last_timestamp_(std::chrono::steady_clock::now()),
    omni_target_priority_{ArmorPriority::fifth}
{
    auto yaml = yaml_load(config_path);
    enemy_color_ = (yaml_read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue; 
    min_detect_count_ = yaml_read<int>(yaml, "min_detect_count");
    max_temp_lost_count_ = yaml_read<int>(yaml, "max_temp_lost_count");
    outpost_max_temp_lost_count_ = yaml_read<int>(yaml, "outpost_max_temp_lost_count");
    normal_temp_lost_count_ = max_temp_lost_count_;

    // 从配置文件读取相机矩阵并计算图像中心坐标
    auto camera_matrix_data = yaml_read<std::vector<double>>(yaml, "camera_matrix");
    double cx = camera_matrix_data[2]; // 相机矩阵的第3个元素为图像中心x坐标
    double cy = camera_matrix_data[5]; // 相机矩阵的第6个元素为图像中心y坐标
    img_center_ = cv::Point2f(cx, cy);
}

std::string Tracker::state() const { return state_; }

std::vector<Target> Tracker::track(
    std::vector<Armor> & armors, std::chrono::steady_clock::time_point t, int id, bool use_enemy_color)
{
    auto dt = delta_time(t, last_timestamp_);
    last_timestamp_ = t;
    set_enemy_color(id);

    // 时间间隔过长，说明可能发生了相机离线
    if (state_ != "lost" && dt > 0.1) {
        logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
        state_ = "lost";
    }
    // 过滤掉非我方装甲板
    armors.erase(
        std::remove_if(armors.begin(), armors.end(), [&](const Armor & a) { return a.color != enemy_color_; }),
        armors.end());

    // 过滤前哨站顶部装甲板
    // armors.erase(std::remove_if(armors.begin(), armors.end(), [this](const Armor & a) {
    //   return a.name == ArmorName::outpost &&
    //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
    //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
    // }), armors.end());

    // 优先选择靠近图像中心的装甲板
    std::sort(armors.begin(), armors.end(), [this](const Armor & a, const Armor & b) {
        auto distance_1 = cv::norm(a.center - img_center_);
        auto distance_2 = cv::norm(b.center - img_center_);
        return distance_1 < distance_2;
    });

    // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
    std::stable_sort(
        armors.begin(), armors.end(),
        [](const Armor & a, const Armor & b) { return a.priority < b.priority; });

    bool found;
    if (state_ == "lost") {
        found = set_target(armors, t);
    }

    else {
        found = update_target(armors, t);
    }

    state_machine(found);

    // 发散检测
    if (state_ != "lost" && target_.diverged()) {
        logger()->debug("[Tracker] Target diverged!");
        state_ = "lost";
        return {};
    }

    // 收敛效果检测：
    if (
        std::accumulate(
            target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
        (0.4 * target_.ekf().window_size)) {
        logger()->debug("[Target] Bad Converge Found!");
        state_ = "lost";
        return {};
    }

    if (state_ == "lost") return {};

    std::vector<Target> targets = {target_};
    return targets;
}

std::tuple<DetectionResult, std::vector<Target>> Tracker::track(
    const std::vector<DetectionResult> & detection_queue, std::vector<Armor> & armors,
    std::chrono::steady_clock::time_point t,int id, bool use_enemy_color)
{
    DetectionResult switch_target{std::vector<Armor>(), t, 0, 0};
    DetectionResult temp_target{std::vector<Armor>(), t, 0, 0};
    if (!detection_queue.empty()) {
        temp_target = detection_queue.front();
    }

    auto dt = delta_time(t, last_timestamp_);
    last_timestamp_ = t;

    // 时间间隔过长，说明可能发生了相机离线
    if (state_ != "lost" && dt > 0.1) {
        logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
        state_ = "lost";
    }

    // 优先选择靠近图像中心的装甲板
    std::sort(armors.begin(), armors.end(), [this](const Armor & a, const Armor & b) {
        auto distance_1 = cv::norm(a.center - img_center_);
        auto distance_2 = cv::norm(b.center - img_center_);
        return distance_1 < distance_2;
    });

    // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
    std::stable_sort(
        armors.begin(), armors.end(),
        [](const Armor & a, const Armor & b) { return a.priority < b.priority; });

    bool found;
    if (state_ == "lost") {
        found = set_target(armors, t);
    }

    // 此时主相机画面中出现了优先级更高的装甲板，切换目标
    else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
        found = set_target(armors, t);
        logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
    }

    // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
    else if (
        state_ == "tracking" && !temp_target.armors.empty() &&
        temp_target.armors.front().priority < target_.priority && target_.convergened()) {
        state_ = "switching";
        switch_target = DetectionResult{
            temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
        omni_target_priority_ = temp_target.armors.front().priority;
        found = false;
        logger()->debug("omniperception find higher priority target");
    }

    else if (state_ == "switching") {
        found = !armors.empty() && armors.front().priority == omni_target_priority_;
    }

    else if (state_ == "detecting" && pre_state_ == "switching") {
        found = set_target(armors, t);
    }

    else {
        found = update_target(armors, t);
    }

    pre_state_ = state_;
    // 更新状态机
    state_machine(found);

    // 发散检测
    if (state_ != "lost" && target_.diverged()) {
        logger()->debug("[Tracker] Target diverged!");
        state_ = "lost";
        return {switch_target, {}};  // 返回switch_target和空的targets
    }

    if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

    std::vector<Target> targets = {target_};
    return {switch_target, targets};
}

void Tracker::state_machine(bool found)
{
    if (state_ == "lost") {
        if (!found) return;

        state_ = "detecting";
        detect_count_ = 1;
    }

    else if (state_ == "detecting") {
        if (found) {
            detect_count_++;
            if (detect_count_ >= min_detect_count_) state_ = "tracking";
        } else {
            detect_count_ = 0;
            state_ = "lost";
        }
    }

    else if (state_ == "tracking") {
        if (found) return;

        temp_lost_count_ = 1;
        state_ = "temp_lost";
    }

    else if (state_ == "switching") {
        if (found) {
            state_ = "detecting";
        } else {
            temp_lost_count_++;
            if (temp_lost_count_ > 200) state_ = "lost";
        }
    }

    else if (state_ == "temp_lost") {
        if (found) {
            state_ = "tracking";
        } else {
            temp_lost_count_++;
            if (target_.name == ArmorName::outpost)
                //前哨站的temp_lost_count需要设置的大一些
                max_temp_lost_count_ = outpost_max_temp_lost_count_;
            else
                max_temp_lost_count_ = normal_temp_lost_count_;

            if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
        }
    }
}

bool Tracker::set_target(std::vector<Armor> & armors, std::chrono::steady_clock::time_point t)
{
    if (armors.empty()) return false;

    auto & armor = armors.front();
    solver_.solve(armor);

    // 根据兵种优化初始化参数
    auto is_balance = (armor.type == ArmorType::big) &&
                                        (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                                            armor.name == ArmorName::five);

    if (is_balance) {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
        target_ = Target(armor, t, 0.2, 2, P0_dig);
    }

    else if (armor.name == ArmorName::outpost) {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
        target_ = Target(armor, t, 0.2765, 3, P0_dig);
    }

    else if (armor.name == ArmorName::base) {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
        target_ = Target(armor, t, 0.3205, 3, P0_dig);
    }

    else {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
        target_ = Target(armor, t, 0.2, 4, P0_dig);
    }

    return true;
}

bool Tracker::update_target(std::vector<Armor> & armors, std::chrono::steady_clock::time_point t)
{
    target_.predict(t);

    int found_count = 0;
    double min_x = 1e10;  // 画面最左侧
    for (const auto & armor : armors) {
        if (armor.name != target_.name || armor.type != target_.armor_type) continue;
        found_count++;
        min_x = armor.center.x < min_x ? armor.center.x : min_x;
    }

    if (found_count == 0) return false;

    for (auto & armor : armors) {
        if (
            armor.name != target_.name || armor.type != target_.armor_type
            //  || armor.center.x != min_x
        )
            continue;

        solver_.solve(armor);

        target_.update(armor);
    }

    return true;
}

bool Tracker::set_enemy_color(int id){
    //红方:1-11 蓝方:101-111
    if ((id < 1 || id > 11) && (id < 101 || id > 111)) {
        // 仅在 id 变化时提示一次，避免刷屏（模拟模式下 id 恒为 0）
        static int last_invalid_id = -1;
        if (id != last_invalid_id) {
            logger()->warn("[Tracker] Invalid ID {}. Using enemy color from config file.", id);
            last_invalid_id = id;
        }
        return false; // 无效的id
    }
    if(id<100){
        enemy_color_ = Color::blue;
    }
    else{
        enemy_color_ = Color::red;
    }
    return true;
}

    Color Tracker::get_enemy_color() const{
        return enemy_color_;
    }

