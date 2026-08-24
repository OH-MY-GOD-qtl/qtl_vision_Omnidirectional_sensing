#ifndef TOOLS__THREAD_POOL_HPP
#define TOOLS__THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>


// 通用线程池：不依赖任何业务类型
class ThreadPool
{
public:
    ThreadPool(size_t num_threads) : stop(false)
    {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
            tasks = std::queue<std::function<void()>>();
        }
        condition.notify_all();
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // 添加任务到任务队列
    template <class F>
    void enqueue(F && f)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

private:
    std::vector<std::thread> workers;         // 工作线程
    std::queue<std::function<void()>> tasks;  // 任务队列
    std::mutex queue_mutex;                   // 任务队列互斥锁
    std::condition_variable condition;        // 条件变量，用于等待任务
    bool stop;                                // 是否停止线程池
};

#endif  // TOOLS__THREAD_POOL_HPP
