#ifndef AUTO_AIM__TENSORRT_INFER_HPP
#define AUTO_AIM__TENSORRT_INFER_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>


// 主机侧张量描述：输出数据统一为 float32
struct TrtTensor
{
    std::string name;
    std::vector<int64_t> dims;  // 完整维度（含 batch 维）
    std::vector<float> data;    // 主机侧数据（kHALF 输出会被转成 float）
};


// TensorRT 引擎 RAII 封装：加载 .engine、复用 context 与显存/主机缓冲。
// 当前约束：静态形状、单输入、FP32 输入；输出支持 FP32 / FP16。
class TensorRTInfer
{
public:
    explicit TensorRTInfer(const std::string & engine_path);
    ~TensorRTInfer();

    TensorRTInfer(const TensorRTInfer &) = delete;
    TensorRTInfer & operator=(const TensorRTInfer &) = delete;

    bool is_ok() const { return ok_; }

    // 输入：主机 NCHW float32（长度 = 输入体积）。
    // 返回：输出张量（引用内部缓冲，下一次 infer 前有效）。
    std::vector<TrtTensor> & infer(const float * input_nchw);

    // 输入张量信息，供调用方预分配输入缓冲
    const TrtTensor & input_info() const { return input_info_; }

private:
    void release();

    nvinfer1::IRuntime * runtime_ = nullptr;
    nvinfer1::ICudaEngine * engine_ = nullptr;
    nvinfer1::IExecutionContext * context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    TrtTensor input_info_;
    std::vector<TrtTensor> outputs_;                        // 输出主机缓冲（复用）
    std::vector<nvinfer1::DataType> output_dtypes_;
    std::vector<void *> input_dev_;                         // 输入显存
    std::vector<void *> output_dev_;                        // 输出显存
    std::vector<std::vector<unsigned char>> half_buffers_;  // kHALF 输出的中转缓冲

    bool ok_ = false;
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
    std::size_t input_vol_ = 0;
    std::size_t input_bytes_ = 0;
    std::vector<unsigned char> input_half_;  // kHALF 输入的中转缓冲
};


#endif  // AUTO_AIM__TENSORRT_INFER_HPP
