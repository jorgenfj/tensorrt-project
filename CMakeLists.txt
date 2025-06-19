cmake_minimum_required(VERSION 3.4)
project(TensorRTProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "/opt/tensorrt-rtx") 
if(NOT EXISTS "${TENSORRT_ROOT}/include")
    message(FATAL_ERROR "TensorRT include directory not found: ${TENSORRT_ROOT}/include")
endif()
if(NOT EXISTS "${TENSORRT_ROOT}/lib")
    message(FATAL_ERROR "TensorRT lib directory not found: ${TENSORRT_ROOT}/lib")
endif()

set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARY_DIRS "${TENSORRT_ROOT}/lib")


find_package(CUDA REQUIRED)

include_directories(include
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)

add_executable(engine_builder
     src/engine_builder.cpp
     src/logger.cpp)

target_link_directories(engine_builder PRIVATE
    ${TENSORRT_LIBRARY_DIRS}
)

target_link_libraries(engine_builder
    libtensorrt_rtx.so
    libtensorrt_onnxparser_rtx.so
    ${CUDA_LIBRARIES}
)