#include "comm/gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#include <algorithm>
#include <fmt/format.h>


Gimbal::Gimbal(const std::string & config_path)
{
    auto yaml = yaml_load(config_path);

    // 无硬件模拟模式：不打开串口，固定自瞄模式
    simulate_ = yaml["simulate"] ? yaml["simulate"].as<bool>() : false;
    debug_serial_ = yaml["debug_serial"] ? yaml["debug_serial"].as<bool>() : false;
    if (simulate_) {
        mode_ = GimbalMode::AUTO_AIM;
        logger()->info("[Gimbal] 模拟模式：不打开串口，固定为自瞄模式");
        return;
    }

    auto com_port = yaml_read<std::string>(yaml, "com_port");
    auto baud_rate = yaml_read<int>(yaml, "baud_rate");
    try {
        serial_.setPort(com_port);
        serial_.setBaudrate(baud_rate); 
        serial::Timeout timeout = serial::Timeout::simpleTimeout(100);
        serial_.setTimeout(timeout);
        serial_.open();
    } catch (const std::exception & e) {
        logger()->error("[Gimbal] Failed to open serial: {}", e.what());
        exit(1);
    }

    thread_ = std::thread(&Gimbal::read_thread, this);

    logger()->info("[Gimbal] Initialized successfully.");
}

Gimbal::~Gimbal()
{
    quit_ = true;
    if (thread_.joinable()) thread_.join();
    if (!simulate_) serial_.close();
}

GimbalMode Gimbal::mode() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

GimbalState Gimbal::state() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
    switch (mode) {
        case GimbalMode::IDLE:
            return "IDLE";
        case GimbalMode::AUTO_AIM:
            return "AUTO_AIM";
        case GimbalMode::SMALL_BUFF:
            return "SMALL_BUFF";
        case GimbalMode::BIG_BUFF:
            return "BIG_BUFF";
        default:
            return "INVALID";
    }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto make_q = [](double yaw, double pitch) {
        return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())).normalized();
    };

    // 无历史数据（模拟模式/刚启动）时回退到当前姿态
    if (q_history_.empty()) return make_q(state_.yaw, state_.pitch);

    // 请求时刻不晚于最早样本：返回最早样本姿态
    if (t <= std::get<1>(q_history_.front())) return std::get<0>(q_history_.front());

    // 请求时刻不早于最新样本：返回最新姿态
    if (t >= std::get<1>(q_history_.back())) return std::get<0>(q_history_.back());

    // 二分查找第一个 timestamp >= t 的样本
    auto it = std::lower_bound(
        q_history_.begin(), q_history_.end(), t,
        [](const auto & item, const std::chrono::steady_clock::time_point & value) {
            return std::get<1>(item) < value;
        });

    if (it == q_history_.begin()) return std::get<0>(*it);
    if (it == q_history_.end()) return std::get<0>(*(it - 1));

    const auto & ahead = *it;
    const auto & behind = *(it - 1);

    double dt_total = delta_time(std::get<1>(ahead), std::get<1>(behind));
    if (dt_total <= 1e-6) return std::get<0>(behind);

    double alpha = delta_time(t, std::get<1>(behind)) / dt_total;
    alpha = std::max(0.0, std::min(1.0, alpha));

    return std::get<0>(behind).slerp(alpha, std::get<0>(ahead)).normalized();
}

void Gimbal::send(VisionToGimbal VisionToGimbal)
{
    tx_data_.mode = VisionToGimbal.mode;
    // 云台约定：yaw 俯视逆时针为正（与内部一致）、pitch 向上为正（内部向上为负，故 pitch 取反）
    tx_data_.yaw = VisionToGimbal.yaw;
    tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
    tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
    tx_data_.pitch = -VisionToGimbal.pitch;
    tx_data_.pitch_vel = -VisionToGimbal.pitch_vel;
    tx_data_.pitch_acc = -VisionToGimbal.pitch_acc;
    tx_data_.is_empty = VisionToGimbal.is_empty;
    tx_data_.shoot = VisionToGimbal.shoot;
    tx_data_.crc16 = get_crc16(
        reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

    if (simulate_) {
        log_sim();
        return;
    }

    try {
        serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
        if (debug_serial_) dump_hex("TX", reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    } catch (const std::exception & e) {
        logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
    }
}

void Gimbal::send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc, bool is_empty)
{
    tx_data_.mode = control ? (fire ? 2 : 1) : 0;
    // 云台约定：yaw 俯视逆时针为正（与内部一致）、pitch 向上为正（内部向上为负，故 pitch 取反）
    tx_data_.yaw = yaw;
    tx_data_.yaw_vel = yaw_vel;
    tx_data_.yaw_acc = yaw_acc;
    tx_data_.pitch = -pitch;
    tx_data_.pitch_vel = -pitch_vel;
    tx_data_.pitch_acc = -pitch_acc;
    tx_data_.is_empty = is_empty;
    tx_data_.shoot = fire;
    tx_data_.crc16 = get_crc16(
        reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

    if (simulate_) {
        log_sim();
        return;
    }

    try {
        serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
        if (debug_serial_) dump_hex("TX", reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    } catch (const std::exception & e) {
        logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
    }
}

void Gimbal::log_sim()
{
    // 模拟模式：按约 10Hz 打印控制信息
    static int count = 0;
    if (++count % 10 == 0)
        logger()->info(
            "[Gimbal][sim] mode={} fire={} yaw={:.2f} pitch={:.2f}", tx_data_.mode, tx_data_.shoot,
            tx_data_.yaw * 57.3, tx_data_.pitch * 57.3);
}

void Gimbal::dump_hex(const char * tag, const uint8_t * data, size_t size)
{
    std::string hex;
    hex.reserve(size * 3);
    for (size_t i = 0; i < size; ++i)
        hex += fmt::format("{:02X} ", static_cast<int>(data[i]));
    logger()->info("[Gimbal][{}] {}bytes: {}", tag, size, hex);
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
    try {
        return serial_.read(buffer, size) == size;
    } catch (const std::exception & e) {
        // logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
        return false;
    }
}

void Gimbal::read_thread()
{
    logger()->info("[Gimbal] read_thread started.");
    int error_count = 0;

    while (!quit_) {
        if (error_count > 5000) {
            error_count = 0;
            logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
            reconnect();
            continue;
        }

        if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
            error_count++;
            continue;
        }

        if (rx_data_.head[0] != 'S' || rx_data_.head[1] != 'P') {
            error_count++;  // 这里需要增加错误计数，否则帧头不匹配时不会增加计数
            continue;
        }

        auto t = std::chrono::steady_clock::now();

        if (!read(
                    reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
                    sizeof(rx_data_) - sizeof(rx_data_.head))) {
            error_count++;
            continue;
        }

        // 直接从packed结构体复制值到临时变量，避免绑定问题
        uint16_t temp_crc16 = rx_data_.crc16;
        if (!check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
            logger()->debug("[Gimbal] CRC16 check failed.RX:{}", temp_crc16);
            error_count++;
            continue;
        }

        error_count = 0;
        if (debug_serial_) dump_hex("RX", reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_));
        // 直接从packed结构体复制值到临时变量
        float temp_yaw = rx_data_.yaw;
        float temp_pitch = rx_data_.pitch;
        float temp_yaw_vel = rx_data_.yaw_vel;
        float temp_pitch_vel = rx_data_.pitch_vel;
        float temp_bullet_speed = rx_data_.bullet_speed;
        uint8_t temp_mode = rx_data_.mode;
        int temp_id = rx_data_.id;// 添加id字段

        std::lock_guard<std::mutex> lock(mutex_);

        // 使用临时变量赋值（云台约定 yaw 俯视逆时针为正（与内部一致）、pitch 向上为正，pitch 转为内部约定向上为负）
        state_.yaw = temp_yaw;
        state_.yaw_vel = temp_yaw_vel;
        state_.pitch = -temp_pitch;
        state_.pitch_vel = -temp_pitch_vel;
        state_.bullet_speed = temp_bullet_speed;
        state_.id = temp_id;// 添加id字段
        // 删除弹丸剩余量

        // 记录姿态-时间戳历史，供 q(t) 按相机捕获时刻插值云台姿态（时间戳对齐）
        Eigen::Quaterniond q_now = (Eigen::AngleAxisd(state_.yaw, Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(state_.pitch, Eigen::Vector3d::UnitY())).normalized();
        q_history_.emplace_back(q_now, t);
        while (q_history_.size() > kQHistoryMax) q_history_.pop_front();

        switch (temp_mode) {
            case 0:
                mode_ = GimbalMode::IDLE;
                break;
            case 1:
                mode_ = GimbalMode::AUTO_AIM;
                break;
            case 2:
                mode_ = GimbalMode::SMALL_BUFF;
                break;
            case 3:
                mode_ = GimbalMode::BIG_BUFF;
                break;
            default:
                mode_ = GimbalMode::IDLE;
                logger()->warn("[Gimbal] Invalid mode: {}", temp_mode);
                break;
        }

        // // 打印接收云台信息
        // logger()->debug("[Gimbal] Data received successfully. Mode: {}, Yaw: {:.2f}, Pitch: {:.2f},CRC: {}",
        //                 temp_id, static_cast<int>(temp_mode), temp_yaw * 57.3, temp_pitch * 57.3, temp_crc16);

    }

    logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
    int max_retry_count = 10;
    for (int i = 0; i < max_retry_count && !quit_; ++i) {
        logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}", i + 1, max_retry_count);
        try {
            serial_.close();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } catch (...) {
        }

        try {
            serial_.open();  // 尝试重新打开
            logger()->info("[Gimbal] Reconnected serial successfully.");
            break;
        } catch (const std::exception & e) {
            logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

