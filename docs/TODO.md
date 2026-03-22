# AxSim-GPU 待完善清单

本文档列出**已完成进度**与**仍需补充**的部分，按优先级和模块分类。

---

## 零、近期已完成（进度快照）

| 项 | 说明 |
|----|------|
| 双网表误差流程 | 已验证：`ref = run_gpu_simulation_only(soa_golden, config)`，`run_gpu_metrics(soa_approx, ref, config)`，同一 `seed` 下 PI 对齐。`main_axsim.cpp` 示例：AND 为 golden、OR（德摩根 AIG）为 approximate。 |
| MRED kernel | `mred_sum_kernel` 已改为**按输出比特**累加 \(\|g-f\|/\max(\|f\|,\delta)\)，与 `metrics.py` 一致；布尔对比时若 \(\delta\) 过小会在 \(f=0\) 时放大，示例中建议 **`delta = 1.0f`**。 |
| 构建 | Makefile：`CUDA_HOME`、`nvcc` 默认 **C++14**、`-I/-L`、DEBUG 下不丢失 `-arch`；`circuit_soa.hpp` 补充 `<tuple>`。 |
| 示例 | 已**不再**用手写 `uint64` 冒充 ref；跨电路对比取代“自比”演示。 |

---

## 一、必须完成（影响正确性 / 可用性）

### 1. 参考输出 `ref_outputs` 的来源

- **现状**：`run_gpu_metrics(soa_approx, ref_outputs, config)` 要求调用方传入已打包好的 `ref_outputs`。**手动路径已打通**：对 golden SoA 调用 `run_gpu_simulation_only` 即可得到与 `seed` 一致的参考比特（见 `main_axsim.cpp`）。
- **仍待做**：
  - **库内封装**（可选）：例如 `run_gpu_metrics_with_reference_circuit(soa_approx, soa_golden, config)`，内部对 `soa_golden` 跑一次 `run_gpu_simulation_only` 再比较，减少重复与误用。
  - 从**文件/ABC** 读入 netlist 后再走上述流程（与第 3 项一起）。

### 2. 大规模 pattern 时的 grid 上限

- **现状**：`nblk = (num_blocks_64 + BLOCK - 1) / BLOCK`，若 `nblk > 65535` 被截断为 65535，导致只仿真了部分 64-pattern 块，其余 pattern 未仿真。
- **待做**：用**多轮 launch** 或 **2D grid** 覆盖全部 `num_blocks_64`，并在 kernel 里用组合 `blockIdx` 计算全局 `block_idx`。

### 3. ABC（或其它 AIG 源）对接

- **现状**：SoA 仍通过 `flatten_from_aig(num_pis, and_nodes, out_ids)` 从内存列表构建；**无**从 AIG 文件或 ABC 直接读入的完整示例。
- **待做**：AIGER 解析或 ABC API；拓扑序 AND 列表；可选 CLI `axsim_main <exact.aig> <approx.aig> -p 65536`。

---

## 二、建议完成（性能与可维护性）

### 4. 输出抽取改为 GPU kernel

- **现状**：在 host 上对每个 output 调用 `cudaMemcpy2D` 把 `node_values` 中对应节点的 strided 列拷到 `d_approx_out`。
- **待做**：`gather_outputs_kernel`：从 `node_values[block_idx * num_nodes + output_node_ids[o]]` 写到 `d_approx_out[o * num_blocks_64 + block_idx]`。

### 5. CUDA 错误检查

- **现状**：`cudaMalloc` / `cudaMemcpy` / `cudaFree` 及 kernel launch 的返回值未检查。
- **待做**：统一宏或 `cudaError_t` 检查，失败时打印 `cudaGetErrorString(err)`。

### 6. Reduction 使用 CUB

- **现状**：Error count / MRED sum 使用自定义 block reduce + host 求和。
- **待做**：`cub::DeviceReduce::Sum` 等全局 reduce（CUDA Toolkit 自带 CUB）。

---

## 三、可选优化与扩展

### 7. Shared memory 缓存 PI

- **待做**：`__shared__` 缓存当前 block 的 PI 行，降低 global 读。

### 8. 随机数：LCG → curand

- **待做**：可选 cuRAND 提升统计性质或可控性。

### 9. PI 生成的 coalesce（蒙特卡洛）

- **现状**：`monte_carlo_kernel_simple` 中仍有 “TODO: coalesce PI generation with other threads” 注释。
- **待做**：warp 协作或结合 shared PI 缓存。

### 10. MEM（Maximum Error Magnitude）

- **待做**：多比特输出时 \(\max_x |g(x)-f(x)|\)。

### 11. 多值输出的 MRED/MSE

- **待做**：按整数输出位宽聚合后再算相对误差 / 平方误差（与当前按 bit 打包的语义区分）。

---

## 四、工程与文档

### 12. 示例与测试

- **现状**：`main_axsim.cpp` 已演示 **两 SoA、golden vs approximate**（AND / OR），非零 ER/MSE；**尚未**有从 AIG 读入的示例。
- **待做**：AIG 示例（依赖第 3 项）；小电路 + 与 `metrics.py` 或解析解的**单元测试 / CI**。

### 13. 文档与 README

- **现状**：README 已增加 **Current status**、Make/CUDA 说明、MRED/δ 提示；**TODO** 本文档已同步进度。
- **待做**：ABC/AIG 流程落地后，补充“从文件运行”与命令行示例；MEM 或多值 metrics 若实现则更新表格。

---

## 汇总表

| 序号 | 项 | 状态 | 说明 |
|------|--------------|------|------|
| 1 | ref 来源 | **部分完成** | 手动两 SoA + `run_gpu_simulation_only` 已验证；库内一站式接口仍可选 |
| 2 | 大规模 pattern | 待做 | 多轮/2D grid |
| 3 | ABC/AIG 对接 | 待做 | 从文件或 ABC 构建 SoA |
| 4 | 输出 gather kernel | 待做 | 替代多次 `cudaMemcpy2D` |
| 5 | CUDA 错误检查 | 待做 | 全部 cuda API |
| 6 | CUB reduction | 待做 | device 全局 reduce |
| 7 | Shared memory PI | 可选 | |
| 8 | curand | 可选 | |
| 9 | PI coalesce | 可选 | |
| 10 | MEM | 可选 | |
| 11 | 多值 MRED/MSE | 可选 | |
| 12 | 示例与测试 | **部分完成** | 跨电路 demo 已有；AIG + 单元测试仍缺 |
| 13 | 文档 | **部分完成** | README 已更新；随 AIG 再补 |

完成 **2、3** 后，库即可在**真实 AIG 文件**流程中完整使用；**1** 的库内封装为体验优化；**4、5、6** 提升健壮性与性能；其余按需求选做。
