#ifndef TOOLS__PLOTTER_HPP
#define TOOLS__PLOTTER_HPP

#include <netinet/in.h>  // sockaddr_in

#include <mutex>
#include <nlohmann/json.hpp>
#include <string>


class Plotter
{
public:
    Plotter(std::string host = "127.0.0.1", uint16_t port = 9870);

    ~Plotter();

    void plot(const nlohmann::json & json);

private:
    int socket_;
    sockaddr_in destination_;
    std::mutex mutex_;
};


#endif  // TOOLS__PLOTTER_HPP