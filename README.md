## Usage

The `engine_builder` executable takes the path to your ONNX model as the first argument and an output path for the generated TensorRT engine.

```bash
./engine_builder <path_to_yolov8.onnx_model> [output_engine_path]
```

## Build Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jorgenfj/tensorrt-project.git
    cd tensorrt-project
    ```

2.  **Ensure your `CMakeLists.txt` is correctly configured:**
    Verify that paths to TensorRT-RTX, CUDA, cuDNN, and OpenCV (if used) are accurate. Specifically, confirm the custom library names for TensorRT-RTX:
    `tensorrt_rtx` and `tensorrt_onnxparser_rtx`.

3.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

4.  **Configure CMake:**
    ```bash
    cmake ..
    ```
    * **Troubleshooting `cmake ..`:**
        * If `TensorRT` libraries are not found, double-check `TENSORRT_ROOT` in `CMakeLists.txt` and ensure `libtensorrt_rtx.so`, `libtensorrt_onnxparser_rtx.so` exist in `${TENSORRT_ROOT}/lib/`.
        * Ensure your `LD_LIBRARY_PATH` environment variable includes your TensorRT and CUDA library paths.

5.  **Build the executable:**
    ```bash
    make
    ```
    * This will compile the project and create the executable named `engine_builder` in the `build/` directory.
