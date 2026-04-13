cmake_minimum_required(VERSION 3.18)
project(AxSim-GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# CUDA
set(CMAKE_CUDA_STANDARD 17)
# Optional: set architecture, e.g. -DCMAKE_CUDA_ARCHITECTURES=70
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90 CACHE STRING "CUDA architectures")
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CPU SoA (no CUDA)
add_library(axsim_soa STATIC src/circuit_soa.cpp src/pattern_file.cpp)
target_include_directories(axsim_soa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA simulation + metrics (contains run_gpu_metrics, run_gpu_simulation_only)
add_library(axsim_cuda STATIC cuda/sim_kernels.cu)
target_include_directories(axsim_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(axsim_cuda PUBLIC CUDA::cudart_static)
set_target_properties(axsim_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

# CPU AIG Monte Carlo (same semantics as GPU kernels)
add_library(axsim_cpu STATIC src/cpu_sim_metrics.cpp)
target_include_directories(axsim_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(axsim_main src/main_axsim.cpp)
target_link_libraries(axsim_main PRIVATE axsim_soa axsim_cuda axsim_cpu)

# Optional: CUB for faster reduction (NVIDIA CUB is in toolkit)
# target_include_directories(axsim_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# Then in .cu: #include <cub/cub.cuh> and use cub::DeviceReduce::Sum
