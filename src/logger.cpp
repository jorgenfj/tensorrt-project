#include "logger.hpp"

void Logger::log(Severity severity, const char* msg) noexcept
{
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

Logger logger;