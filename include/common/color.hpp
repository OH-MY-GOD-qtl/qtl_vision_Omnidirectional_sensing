#ifndef COMMON__COLOR_HPP
#define COMMON__COLOR_HPP

#include <string>
#include <vector>


// 装甲板颜色（被 armor/lightbar/tracker 等模块共用，独立成头避免循环依赖）
enum Color
{
    red,
    blue,
    extinguish,
    purple
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};


#endif  // COMMON__COLOR_HPP
