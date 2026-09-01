#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "common/common.hpp"
#include "serial/serial.h"



// 云台到视觉通信数据结构体
struct __attribute__((packed)) GimbalToVision
{
    uint8_t head[2] = {'S', 'P'};
    uint8_t mode;  // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
    float yaw;
    float yaw_vel;
    float pitch;
    float pitch_vel;
    float bullet_speed;  // 子弹速度
    int id;
    uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) <= 64);
// 视觉到云台通信数据结构体
struct __attribute__((packed)) VisionToGimbal
{
    uint8_t head[2] = {'S', 'P'};
    uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
    float yaw;
    float yaw_vel;
    float yaw_acc;
    float pitch;
    float pitch_vel;
    float pitch_acc;
    bool shoot;
    bool is_empty;  // 目标是否为空
    uint16_t crc16;
};

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
    IDLE,        // 空闲
    AUTO_AIM,    // 自瞄
    SMALL_BUFF,  // 小符
    BIG_BUFF     // 大符
};

struct GimbalState
{
    float yaw = 0;
    float yaw_vel = 0;
    float pitch = 0;
    float pitch_vel = 0;
    float bullet_speed = DEFAULT_BULLET_SPEED;
    int id = 0;
};

class Gimbal
{
public:
    Gimbal(const std::string & config_path);

    ~Gimbal();

    GimbalMode mode() const;
    GimbalState state() const;
    std::string str(GimbalMode mode) const;
    Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

    void send(
        bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
        float pitch_acc, bool is_empty = true);

    void send(VisionToGimbal VisionToGimbal);

private:
    bool simulate_ = false;  // 无硬件模拟模式：不打开串口，固定自瞄
    bool debug_serial_ = false;  // 打印串口收发原始字节（调试用）
    serial::Serial serial_;

    std::thread thread_;
    std::atomic<bool> quit_ = false;
    mutable std::mutex mutex_;

    GimbalToVision rx_data_;
    VisionToGimbal tx_data_;

    GimbalMode mode_ = GimbalMode::IDLE;
    GimbalState state_;
    // 姿态-时间戳历史缓冲：按相机捕获时刻插值云台姿态（时间戳对齐）
    std::deque<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>> q_history_;
    static constexpr size_t kQHistoryMax = 500;  // 约0.5s @1kHz（按实际串口频率自适应）
    bool read(uint8_t * buffer, size_t size);
    void read_thread();
    void reconnect();
    void log_sim();  // 模拟模式下的控制信息打印
    void dump_hex(const char * tag, const uint8_t * data, size_t size);
};


#endif  // IO__GIMBAL_HPP