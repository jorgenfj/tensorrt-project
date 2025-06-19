#pragma once

#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;


class Logger : public ILogger
{
public:
    Logger() = default;

    void log(Severity severity, const char* msg) noexcept override;
};

extern Logger logger;