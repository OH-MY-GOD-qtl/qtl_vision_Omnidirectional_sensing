#include "detector/tensorrt_infer.hpp"

#include <cuda_fp16.h>

#include <cstdio>
#include <fstream>
#include <utility>


namespace
{
// 文件级 logger：把 TensorRT 内部日志转发到 stderr（与项目 spdlog 解耦）
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char * msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

// 维度体积；含动态维（-1）时返回 -1
inline int64_t volume(const nvinfer1::Dims & d)
{
    if (d.nbDims <= 0) return 0;
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) return -1;
        v *= d.d[i];
    }
    return v;
}

inline std::size_t elem_size(nvinfer1::DataType dt)
{
    switch (dt) {
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kFLOAT:
        default: return 4;
    }
}

inline std::vector<int64_t> dims_to_vec(const nvinfer1::Dims & d)
{
    std::vector<int64_t> v;
    v.reserve(d.nbDims);
    for (int64_t i = 0; i < d.nbDims; ++i) v.push_back(d.d[i]);
    return v;
}
}  // namespace


TensorRTInfer::TensorRTInfer(const std::string & engine_path)
{
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "[TensorRT] 无法打开引擎文件: %s\n", engine_path.c_str());
        return;
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    if (size <= 0 || !file.read(blob.data(), size)) {
        std::fprintf(stderr, "[TensorRT] 读取引擎文件失败: %s\n", engine_path.c_str());
        return;
    }

    static TrtLogger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        std::fprintf(stderr, "[TensorRT] 创建 runtime 失败\n");
        return;
    }
    engine_ = runtime_->deserializeCudaEngine(blob.data(), size);
    if (!engine_) {
        std::fprintf(stderr, "[TensorRT] 反序列化引擎失败\n");
        return;
    }
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::fprintf(stderr, "[TensorRT] 创建执行上下文失败\n");
        return;
    }

    // 枚举输入 / 输出张量
    std::vector<std::string> in_names, out_names;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char * name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            in_names.push_back(name);
        } else {
            out_names.push_back(name);
        }
    }
    if (in_names.size() != 1) {
        std::fprintf(stderr, "[TensorRT] 当前仅支持单输入模型，实际输入数=%zu\n", in_names.size());
        return;
    }

    // 输入：要求 FP32 / FP16 + 静态形状
    auto in_dims = engine_->getTensorShape(in_names[0].c_str());
    auto in_dtype = engine_->getTensorDataType(in_names[0].c_str());
    if (in_dtype != nvinfer1::DataType::kFLOAT && in_dtype != nvinfer1::DataType::kHALF) {
        std::fprintf(stderr, "[TensorRT] 不支持的输入类型（仅支持 FP32/FP16），实际=%d\n",
            static_cast<int>(in_dtype));
        return;
    }
    input_dtype_ = in_dtype;
    const int64_t in_vol = volume(in_dims);
    if (in_vol <= 0) {
        std::fprintf(stderr, "[TensorRT] 输入形状含动态维或非法\n");
        return;
    }
    input_vol_ = in_vol;
    context_->setInputShape(in_names[0].c_str(), in_dims);

    input_info_.name = in_names[0];
    input_info_.dims = dims_to_vec(in_dims);

    // 输出：记录形状 / 类型并预分配显存 + 主机缓冲
    for (const auto & on : out_names) {
        auto dims = context_->getTensorShape(on.c_str());
        auto dtype = engine_->getTensorDataType(on.c_str());
        const int64_t vol = volume(dims);
        if (vol <= 0) {
            std::fprintf(stderr, "[TensorRT] 输出 %s 形状非法\n", on.c_str());
            return;
        }

        TrtTensor t;
        t.name = on;
        t.dims = dims_to_vec(dims);
        t.data.assign(vol, 0.0f);
        outputs_.push_back(std::move(t));
        output_dtypes_.push_back(dtype);
        half_buffers_.push_back({});

        void * dev = nullptr;
        if (cudaMalloc(&dev, vol * elem_size(dtype)) != cudaSuccess) {
            std::fprintf(stderr, "[TensorRT] cudaMalloc 输出失败\n");
            return;
        }
        output_dev_.push_back(dev);
    }

    // 输入显存
    input_bytes_ = input_vol_ * elem_size(input_dtype_);
    if (input_dtype_ == nvinfer1::DataType::kHALF) {
        input_half_.resize(input_vol_ * 2);
    }
    void * dev_in = nullptr;
    if (cudaMalloc(&dev_in, input_bytes_) != cudaSuccess) {
        std::fprintf(stderr, "[TensorRT] cudaMalloc 输入失败\n");
        return;
    }
    input_dev_.push_back(dev_in);

    // 绑定输入/输出张量地址（enqueueV3 要求先 setTensorAddress；固定缓冲复用，只需设置一次）
    context_->setTensorAddress(input_info_.name.c_str(), input_dev_[0]);
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
        context_->setTensorAddress(outputs_[i].name.c_str(), output_dev_[i]);
    }

    cudaStreamCreate(&stream_);
    ok_ = true;
}


TensorRTInfer::~TensorRTInfer()
{
    release();
}


void TensorRTInfer::release()
{
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    for (void * p : input_dev_) {
        if (p) cudaFree(p);
    }
    for (void * p : output_dev_) {
        if (p) cudaFree(p);
    }
    input_dev_.clear();
    output_dev_.clear();
    // TensorRT 10+：用 delete 释放对象（destroy() 已移除）
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}


std::vector<TrtTensor> & TensorRTInfer::infer(const float * input_nchw)
{
    // 输入精度为 FP16 时，先把主机 float 转成 half（与引擎输入 dtype 对齐）
    const void * h2d_ptr = input_nchw;
    if (input_dtype_ == nvinfer1::DataType::kHALF) {
        __half * dst = reinterpret_cast<__half *>(input_half_.data());
        const float * src = input_nchw;
        for (std::size_t i = 0; i < input_vol_; ++i) dst[i] = __float2half(src[i]);
        h2d_ptr = input_half_.data();
    }

    // H2D 输入
    cudaMemcpyAsync(input_dev_[0], h2d_ptr, input_bytes_, cudaMemcpyHostToDevice, stream_);

    if (!context_->enqueueV3(stream_)) {
        std::fprintf(stderr, "[TensorRT] enqueueV3 执行失败\n");
    }

    // D2H 输出
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
        const std::size_t vol = outputs_[i].data.size();
        if (output_dtypes_[i] == nvinfer1::DataType::kFLOAT) {
            cudaMemcpyAsync(
                outputs_[i].data.data(), output_dev_[i], vol * sizeof(float),
                cudaMemcpyDeviceToHost, stream_);
        } else if (output_dtypes_[i] == nvinfer1::DataType::kHALF) {
            half_buffers_[i].resize(vol * 2);
            cudaMemcpyAsync(
                half_buffers_[i].data(), output_dev_[i], vol * 2, cudaMemcpyDeviceToHost, stream_);
        }
    }

    cudaStreamSynchronize(stream_);

    // kHALF 输出转成 float
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
        if (output_dtypes_[i] == nvinfer1::DataType::kHALF) {
            const __half * src = reinterpret_cast<const __half *>(half_buffers_[i].data());
            float * dst = outputs_[i].data.data();
            for (std::size_t j = 0; j < outputs_[i].data.size(); ++j) {
                dst[j] = __half2float(src[j]);
            }
        }
    }

    return outputs_;
}
