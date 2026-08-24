#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP


struct BallisticTrajectory
{
    bool unsolvable;
    double fly_time;
    double pitch;  // 抬头为正

    // 不考虑空气阻力
    // v0 子弹初速度大小，单位：m/s
    // d 目标水平距离，单位：m
    // h 目标竖直高度，单位：m
    BallisticTrajectory(const double v0, const double d, const double h);
};


#endif  // TOOLS__TRAJECTORY_HPP