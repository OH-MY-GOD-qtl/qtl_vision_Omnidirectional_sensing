#ifndef TOOLS__EXTENDED_KALMAN_FILTER_HPP
#define TOOLS__EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>


class ExtendedKalmanFilter
{
public:
    Eigen::VectorXd x;
    Eigen::MatrixXd P;

    ExtendedKalmanFilter() = default;

    ExtendedKalmanFilter(
        const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
            [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

    // 使用 Eigen::Ref 使固定尺寸矩阵（如 Matrix<double,11,11>）可零拷贝传入
    Eigen::VectorXd predict(
        const Eigen::Ref<const Eigen::MatrixXd> & F, const Eigen::Ref<const Eigen::MatrixXd> & Q);

    Eigen::VectorXd predict(
        const Eigen::Ref<const Eigen::MatrixXd> & F, const Eigen::Ref<const Eigen::MatrixXd> & Q,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

    Eigen::VectorXd update(
        const Eigen::Ref<const Eigen::VectorXd> & z, const Eigen::Ref<const Eigen::MatrixXd> & H,
        const Eigen::Ref<const Eigen::MatrixXd> & R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
            [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

    Eigen::VectorXd update(
        const Eigen::Ref<const Eigen::VectorXd> & z, const Eigen::Ref<const Eigen::MatrixXd> & H,
        const Eigen::Ref<const Eigen::MatrixXd> & R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
            [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

    std::map<std::string, double> data;  //卡方检验数据
    std::deque<int> recent_nis_failures{0};
    size_t window_size = 100;
    double last_nis;

private:
    Eigen::MatrixXd I;
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;

    int nees_count_ = 0;
    int nis_count_ = 0;
    int total_count_ = 0;
};


#endif  // TOOLS__EXTENDED_KALMAN_FILTER_HPP