#include "tools/exiter.hpp"

#include <csignal>
#include <stdexcept>


bool exit_ = false;
bool exiter_inited_ = false;

Exiter::Exiter()
{
    if (exiter_inited_) throw std::runtime_error("Multiple Exiter instances!");
    std::signal(SIGINT, [](int) {
        if (exit_) {
            // 第二次 Ctrl+C：主线程可能阻塞在 read() 等位置无法退出，恢复默认处理强制终止
            std::signal(SIGINT, SIG_DFL);
            std::raise(SIGINT);
        }
        exit_ = true;
    });
    exiter_inited_ = true;
}

bool Exiter::exit() const { return exit_; }

