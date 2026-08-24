#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "buff/buff_detector.hpp"
#include "buff/buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/ransac_sine_fitter.hpp"


class Voter
{
public:
    Voter();
    void vote(const double angle_last, const double angle_now);
    int clockwise();

private:
    int clockwise_;
};

/// BuffTarget 基类

class BuffTarget
{
public:
    BuffTarget();
    virtual void get_target(
        const std::optional<PowerRune> & p,
        std::chrono::steady_clock::time_point & timestamp) = 0;  // 纯虚函数

    virtual void predict(double dt) = 0;  // 纯虚函数

    Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

    bool is_unsolve() const;

    Eigen::VectorXd ekf_x() const;

    double spd = 0;  //调试用

protected:
    virtual void init(double nowtime, const PowerRune & p) = 0;  // 纯虚函数

    virtual void update(double nowtime, const PowerRune & p) = 0;  // 纯虚函数

    Eigen::VectorXd x0_;
    Eigen::MatrixXd P0_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd H_;
    Eigen::MatrixXd R_;
    ExtendedKalmanFilter ekf_;
    double lasttime_ = 0;
    Voter voter;  // 逆时针-1 顺时针1
    bool first_in_;
    bool unsolvable_;
};

/// SmallTarget子类

class SmallTarget : public BuffTarget
{
public:
    SmallTarget();

    void get_target(
        const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

    void predict(double dt) override;

private:
    void init(double nowtime, const PowerRune & p) override;

    void update(double nowtime, const PowerRune & p) override;

    Eigen::MatrixXd h_jacobian() const;

    const double SMALL_W = CV_PI / 3;
    // const double SMALL_W = 0;
};

/// BigTarget子类

class BigTarget : public BuffTarget
{
public:
    BigTarget();

    void get_target(
        const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

    void predict(double dt) override;

private:
    void init(double nowtime, const PowerRune & p) override;

    void update(double nowtime, const PowerRune & p) override;

    Eigen::MatrixXd h_jacobian() const;

    RansacSineFitter spd_fitter_;

    double fit_spd_;
};

#endif