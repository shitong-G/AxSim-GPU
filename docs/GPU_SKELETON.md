# AxSim-GPU 代码骨架说明

## 结构概览

- **A. CPU 端内存重排 (SoA)**  
  - `include/axsim/circuit_soa.hpp`：SoA 结构体与 `flatten_from_aig()` 声明  
  - `src/circuit_soa.cpp`：从 AIG 拓扑填充 `fanin0_indices`、`fanin1_indices`、`is_compl0`、`is_compl1`  
  - 你只需在 `circuit_soa.cpp` 中对接 ABC（或其它库）的图结构，按拓扑顺序填入上述数组，然后 `cudaMemcpy` 到设备

- **B. GPU 蒙特卡洛仿真**  
  - `cuda/sim_kernels.cu`  
    - `DeviceCircuitSoA`：设备端 SoA（指针在 host 上设为 device 指针后传入 kernel）  
    - `monte_carlo_kernel` / `monte_carlo_kernel_simple`：每个 thread 负责一个 `uint64_t`（64 个 pattern），按拓扑顺序求 AND，PI 用 LCG 在设备上生成  
  - 可扩展：用 shared memory 预加载当前 block 的 PI 行，减少对 node_values 的全局访问

- **C. Metric 的 Reduction**  
  - `error_count_kernel`：对 `approx ^ reference` 做 `__popcll()`，再 block 内 tree reduce，block 结果写回 `block_sums`，host 上再求和得总 error count，除以 `(num_outputs * num_patterns)` 得 Error Rate  
  - `mred_sum_kernel`：按相对误差公式累加，block reduce，再在 host 求平均得 MRED  
  - `mse_sum_kernel`：对 0/1 输出等价于 error count，MSE = mean squared error  
  - 后续可用 CUB 的 `cub::DeviceReduce::Sum` 做单次大 reduction，替代多 block + host 汇总

## 数据布局

- **node_values**：`[num_blocks_64][num_nodes]`，即每个 64-pattern 块一行，一行内按节点 id 存一个 `uint64_t`  
- **approx/reference**：`[num_outputs][num_blocks_64]`，按输出、再按 64-pattern 块排布，与 `metrics.py` 里按 pattern 展开的比特一致  
- 从 `node_values` 抽输出到 `d_approx_out` 时，用 `cudaMemcpy2D` 按列（每个输出节点一列）从 strided 的 `node_values` 拷到连续的 `d_approx_out`

## 你需要补的细节

1. **ABC 对接**：在 `circuit_soa.cpp`（或新文件）里用 ABC API 遍历 AIG，得到每个 AND 的 fanin id 和取反位，调用 `flatten_from_aig()` 或直接填充 `CircuitSoA`  
2. **参考输出 ref_outputs**：由黄金网表在 CPU 上仿真得到，或由另一份 GPU 仿真得到，格式为 `num_outputs * ceil(num_patterns/64)` 个 `uint64_t`  
3. **Shared memory 优化**：在 `monte_carlo_kernel` 中为当前 block 的 PI 行开 `__shared__ uint64_t pi_cache[]`，先集体加载再算 AND  
4. **CUB**：用 CUB 做单遍 reduction，减少 kernel  launch 和 host 汇总  
5. **多输出 gather**：当前用多次 `cudaMemcpy2D` 抽取输出，可改为一个 kernel：每个 thread 负责若干 (output_id, block_64_id)，从 `node_values` 读到 `d_approx_out`

## 编译与运行

**Make（推荐）：**
```bash
make
./build/axsim_main
```
可选：`make DEBUG=1` 调试版；`make CUDA_ARCH=sm_80` 指定架构；`make run` 直接运行。

**CMake：**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./axsim_main
```

需要已安装 CUDA Toolkit；使用 Make 时需系统有 `make` 和 `g++`，使用 CMake 时需 CMake 3.18+。
