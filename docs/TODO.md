# AxSim-GPU 待完善清单

本文档列出当前代码框架中**需要你补充或修复**的部分，按优先级和模块分类。

---

## 一、必须完成（影响正确性 / 可用性）

### 1. 参考输出 `ref_outputs` 的来源

- **现状**：`run_gpu_metrics(soa, ref_outputs, config)` 要求调用方传入已打包好的 `ref_outputs`（黄金网表在相同 pattern 下的输出比特）。
- **待做**：
  - 若用**黄金 AIG**：对同一批随机 pattern 在 CPU 或 GPU 上跑一次黄金电路仿真，得到 `std::vector<uint64_t> ref_outputs`，再传入。
  - 或提供**库内接口**：例如 `run_gpu_metrics_with_reference_circuit(soa_approx, soa_golden, config)`，内部对 `soa_golden` 跑一次 `run_gpu_simulation_only` 得到 `ref_outputs`，再与近似电路比较。（注意两次仿真需用同一组随机 pattern，即同一 seed 或显式传入同一批 PI。）

### 2. 大规模 pattern 时的 grid 上限

- **现状**：`nblk = (num_blocks_64 + BLOCK - 1) / BLOCK`，若 `nblk > 65535` 被截断为 65535，导致只仿真了 65535×256 个 64-pattern 块，其余 pattern 未仿真。
- **待做**：用**多轮 launch** 或 **2D grid**（如 `dim3 grid((nblk + 65535 - 1) / 65535, 65535)` 等）覆盖全部 `num_blocks_64`，并在 kernel 里用 `blockIdx.x + blockIdx.y * 65535` 等计算全局 block 下标。

### 3. ABC（或其它 AIG 源）对接

- **现状**：SoA 仅能通过 `flatten_from_aig(num_pis, and_nodes, out_ids)` 从内存中的列表构建；没有从 AIG 文件或 ABC 直接读入的代码。
- **待做**：
  - 在 `src/` 下实现 AIG 解析（如读 AIGER 或调用 ABC API），遍历得到拓扑序的 AND 列表及每个 AND 的 fanin id、取反位。
  - 调用现有 `flatten_from_aig()` 或直接填充 `CircuitSoA`，供后续 `run_gpu_metrics` / `run_gpu_simulation_only` 使用。
  - 可选：提供命令行工具，例如 `axsim_main <exact.aig> <approx.aig> -p 65536`，内部完成读 AIG → SoA → 黄金仿真 → 近似仿真 → 输出 ER/MRED/MSE。

---

## 二、建议完成（性能与可维护性）

### 4. 输出抽取改为 GPU kernel

- **现状**：在 host 上对每个 output 调用 `cudaMemcpy2D`，把 `node_values` 中对应节点的 strided 列拷到 `d_approx_out`。
- **待做**：写一个 `gather_outputs_kernel`：每个 thread 负责若干 `(output_id, block_64_id)`，从 `node_values[block_idx * num_nodes + output_node_ids[o]]` 读到 `d_approx_out[o * num_blocks_64 + block_idx]`。需要把 `output_node_ids` 拷到 device 或作为 constant。这样可减少多次 D2D copy 和 launch 开销。

### 5. CUDA 错误检查

- **现状**：`cudaMalloc` / `cudaMemcpy` / `cudaFree` 及 kernel launch 的返回值未检查。
- **待做**：在关键路径上使用 `cudaError_t err = cuda...; if (err != cudaSuccess) { ... }`，或在 Debug 构建中统一用宏包装，失败时打印 `cudaGetErrorString(err)` 并返回错误或 abort。

### 6. Reduction 使用 CUB

- **现状**：Error count / MRED sum 使用自定义 block reduce，结果写回 `block_sums`，再在 host 上循环求和。
- **待做**：用 CUB 的 `cub::DeviceReduce::Sum`（或类似）在 device 上做单次全局 reduction，减少 kernel 数和 host 端循环，代码更简洁且通常更快。需在 `sim_kernels.cu` 中 `#include <cub/cub.cuh>` 并链接/包含 CUB（CUDA 自带）。

---

## 三、可选优化与扩展

### 7. Shared memory 缓存 PI

- **现状**：蒙特卡洛 kernel 中每个 thread 从 global 的 `row[]` 读 fanin 值；PI 的读取频率高。
- **待做**：在 `monte_carlo_kernel` 中为当前 block 的 PI 行分配 `__shared__ uint64_t pi_cache[]`，由 block 内线程协作加载一次，AND 计算时若 fanin 是 PI 则从 shared 读。可减少对 `node_values` 的 global 访问。

### 8. 随机数：LCG → curand

- **现状**：设备上用简单 LCG 生成随机比特，统计性质一般。
- **待做**：若需要更好统计质量或可重复性，可改用 cuRAND（如 `curand_init` + `curand()`），在 kernel 中按 block/thread 初始化状态并生成 PI 的 uint64。会略增依赖与复杂度。

### 9. PI 生成的 coalesce（蒙特卡洛）

- **现状**：`monte_carlo_kernel_simple` 里注释了 “TODO: coalesce PI generation with other threads”；当前每个 thread 独立写自己的 `row[i]`，对 global 的写是合并的，但可进一步优化。
- **待做**：若同一 block 内多个 thread 负责不同 block_idx，可考虑让同一 warp 协作生成/写 PI，或与 shared memory 缓存结合，减少重复计算（若存在）。

### 10. MEM（Maximum Error Magnitude）

- **现状**：README 中列出 MEM 为 “planned”；当前仅实现 ER、MRED、MSE。
- **待做**：若电路输出为多比特整数值，MEM = max_x |g(x) − f(x)|。实现方式：仿真得到近似与参考的整型输出（或从现有比特打包），在 GPU 上求差、取绝对值，再用 `cub::DeviceReduce::Max` 或自定义 max-reduction 得到全局最大值。

### 11. 多值输出的 MRED/MSE

- **现状**：MRED/MSE 的 kernel 按比特差（或 0/1）处理；对多比特整数输出，当前逻辑相当于把每个 bit 当独立 0/1 处理。
- **待做**：若需要“按整数输出”的 MRED/MSE（即先把每 64 个 pattern 的比特聚合成整数再算 |g−f|/max(|f|,δ) 和 (g−f)²），需在 kernel 中按输出位宽打包成整数再算相对误差/平方误差，并相应调整 reduction。

---

## 四、工程与文档

### 12. 示例与测试

- **现状**：`main_axsim.cpp` 仅演示一个 2-input AND，ref 为手写的一个 uint64。
- **待做**：
  - 增加从 AIG 文件读入的示例（在完成第 3 项后）。
  - 加一两组小电路 + 黄金 ref，对 ER/MRED/MSE 与 Python `metrics.py` 或手算结果做数值对比，作为单元测试或 CI。

### 13. 文档与 README

- **现状**：README 已按学术库形式写好；`docs/GPU_SKELETON.md` 描述设计与扩展点。
- **待做**：完成 ABC 对接或 AIG 读取后，在 README 中补充“从 AIG 文件运行”的示例命令和接口说明；若增加 MEM 或多值输出，同步更新 README 的 metrics 表格与 API。

---

## 汇总表

| 序号 | 项           | 类型     | 说明 |
|------|--------------|----------|------|
| 1    | ref_outputs 来源 | 必须     | 黄金电路仿真或库内接口 |
| 2    | 大规模 pattern   | 必须     | 多轮/2D grid 覆盖全部块 |
| 3    | ABC/AIG 对接     | 必须     | 从文件或 ABC 构建 SoA |
| 4    | 输出 gather kernel | 建议   | 替代多次 cudaMemcpy2D |
| 5    | CUDA 错误检查    | 建议     | 所有 cuda* 与 kernel 检查 |
| 6    | CUB reduction    | 建议     | 单次 device reduce |
| 7    | Shared memory PI | 可选     | 降低 global 带宽 |
| 8    | curand           | 可选     | 更好随机数 |
| 9    | PI coalesce      | 可选     | 与 7 可一起做 |
| 10   | MEM 指标         | 可选     | 多比特输出时 max \|g−f\| |
| 11   | 多值 MRED/MSE    | 可选     | 按整数输出计算 |
| 12   | 示例与测试       | 工程     | AIG 示例 + 数值校验 |
| 13   | 文档更新         | 工程     | README/API 与实现同步 |

完成 **1、2、3** 后，库即可在真实 AIG 流程中正确使用；**4、5、6** 能明显提升健壮性和性能；其余按需求选做。
