#include "detector/multithread/commandgener.hpp"

#include "tools/math_tools.hpp"


namespace multithread
{

CommandGener::CommandGener(
    Shooter & shooter, Aimer & aimer, Gimbal & gimbal,
    Plotter & plotter, bool debug)
: shooter_(shooter), aimer_(aimer), gimbal_(gimbal), plotter_(plotter), stop_(false), debug_(debug)
{
    thread_ = std::thread(&CommandGener::generate_command, this);
}

CommandGener::~CommandGener()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void CommandGener::push(
    const std::vector<Target> & targets, const std::chrono::steady_clock::time_point & t,
    double bullet_speed, const Eigen::Vector3d & gimbal_pos)
{
    std::lock_guard<std::mutex> lock(mtx_);
    latest_ = {targets, t, bullet_speed, gimbal_pos};
    cv_.notify_one();
}

void CommandGener::generate_command()
{
    auto t0 = std::chrono::steady_clock::now();
    while (!stop_) {
        std::optional<Input> input;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (latest_ && delta_time(std::chrono::steady_clock::now(), latest_->t) < 0.2) {
                input = latest_;
            } else
                input = std::nullopt;
        }
        if (input) {
            //can通信的命令生成
            // auto command = aimer_.aim(input->targets_, input->t, input->bullet_speed);
            // command.shoot = shooter_.shoot(command, aimer_, input->targets_, input->gimbal_pos);
            // command.horizon_distance = input->targets_.empty()
            //                              ? 0
            //                              : std::sqrt(
            //                                  square(input->targets_.front().ekf_x()[0]) +
            //                                  square(input->targets_.front().ekf_x()[2]));

            //串口gimbal命令生成
            VisionToGimbal VisionToGimbal;
            auto command = aimer_.aim(input->targets_, input->t, input->bullet_speed);
            bool fire = shooter_.shoot(command, aimer_, input->targets_, input->gimbal_pos);
            VisionToGimbal.mode = input->targets_.empty() ? 0 : 2;  //0不控制，2控制且开火
            VisionToGimbal.yaw = command.yaw;
            VisionToGimbal.pitch = command.pitch;  // 符号统一在 Gimbal::send 处理
            VisionToGimbal.shoot = fire;
            VisionToGimbal.is_empty = input->targets_.empty();



            gimbal_.send(VisionToGimbal);  // 串口通信至云台

            //debug VisionToGimbal Data
            // if (debug_) {
            //   int mode = static_cast<int>(VisionToGimbal.mode);
            //   float yaw = VisionToGimbal.yaw;
            //   float pitch = VisionToGimbal.pitch;
            //   bool shoot = VisionToGimbal.shoot;
            //   logger()->debug(
            //     "[CommandGener] Data sent to gimbal. Mode: {}, Yaw: {:.2f}, Pitch: {:.2f}, "
            //     "Shoot: {}",
            //     mode, yaw * 57.3, pitch * 57.3, shoot);
            // }

            if (debug_) {
                nlohmann::json data;
                data["t"] = delta_time(std::chrono::steady_clock::now(), t0);
                int mode = static_cast<int>(VisionToGimbal.mode);
                float yaw = VisionToGimbal.yaw;
                float pitch = VisionToGimbal.pitch;
                bool shoot = VisionToGimbal.shoot;
                data["cmd_mode"] = mode;
                data["cmd_yaw"] = yaw * 57.3;
                data["cmd_pitch"] = pitch * 57.3;
                data["shoot"] = shoot;
                data["is_empty"] = VisionToGimbal.is_empty;
                plotter_.plot(data);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  //approximately 200Hz
    }
}

}  // namespace multithread

