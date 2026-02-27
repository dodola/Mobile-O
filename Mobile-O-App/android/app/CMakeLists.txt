cmake_minimum_required(VERSION 3.22.1)
project("mobileo_native")

# VAE post-processing kernels (equivalent to VAEKernels.metal on iOS)
add_library(
    mobileo_native
    SHARED
    src/main/cpp/vae_postprocess.cpp
)

# Enable ARM NEON optimizations
target_compile_options(mobileo_native PRIVATE
    -O3
    -ffast-math
    -std=c++17
)

find_library(log-lib log)

target_link_libraries(
    mobileo_native
    ${log-lib}
)
