#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP


struct Command
{
    bool control;
    bool shoot;
    double yaw;
    double pitch;
    double horizon_distance = 0;  //无人机专有
};


#endif  // IO__COMMAND_HPP