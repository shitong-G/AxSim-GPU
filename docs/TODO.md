# AxSim-GPU 待完善清单

本文档列出**已完成进度**与**仍需补充**的部分，按优先级和模块分类。

---

## 零、近期已完成（进度快照）

| 项 | 说明 |
|----|------|
| 双网表误差流程 | 已完成 `run_gpu_metrics_pair(soa_approx, soa_golden, config)`：内部生成参考输出并做 PI/PO 一致性校验。 |
| 指标语义与接口 | 已增加 `mae_norm`（并保留 `mred` 兼容别名）、`GpuMetricsResult.ok`、`outputs_msb_first` 位序开关。 |
| 正确性与鲁棒性 | 已补齐 CUDA 错误检查；`num_patterns` 超大时改为分批 launch，避免 65535 grid 截断。 |
| 构建 | Makefile：`CUDA_HOME`、`nvcc` 默认 **C++14**、`-I/-L`、DEBUG 下不丢失 `-arch`；`circuit_soa.hpp` 补充 `<tuple>`。 |
| 示例与测试 | `main_axsim.cpp` 已支持命令行输入（netlist 路径/pattern/seed 等）；`manual_metrics_test.cpp` 已覆盖手算可验证案例。 |

---

## 一、必须完成（影响正确性 / 可用性）

### 1. 参考输出 `ref_outputs` 的来源

- **现状**：已完成。`run_gpu_metrics_pair(soa_approx, soa_golden, config)` 提供一站式双网表评估路径。
- **仍待做**：按需要增加更细粒度的输出映射策略（例如自定义 PO bit map）。

### 2. 大规模 pattern 时的 grid 上限

- **现状**：已完成。改为多轮 launch，完整覆盖全部 `num_blocks_64`。
- **待做**：可选 2D grid 版本与性能对比。

### 3. ABC（或其它 AIG 源）对接

- **现状**：SoA 仍通过 `flatten_from_aig(num_pis, and_nodes, out_ids)` 从内存列表构建；**无**从 AIG 文件或 ABC 直接读入的完整示例。
- **待做**：AIGER 解析或 ABC API；拓扑序 AND 列表；可选 CLI `axsim_main <exact.aig> <approx.aig> -p 65536`。

---

## 二、建议完成（性能与可维护性）

### 4. 输出抽取改为 GPU kernel

- **现状**：在 host 上对每个 output 调用 `cudaMemcpy2D` 把 `node_values` 中对应节点的 strided 列拷到 `d_approx_out`。
- **待做**：`gather_outputs_kernel`：从 `node_values[block_idx * num_nodes + output_node_ids[o]]` 写到 `d_approx_out[o * num_blocks_64 + block_idx]`。

### 5. CUDA 错误检查

- **现状**：已完成。关键 CUDA API 和 kernel launch 已检查并输出错误信息。
- **待做**：按需封装成统一宏，减少样板代码。

### 6. Reduction 使用 CUB

- **现状**：Error count / MRED sum 使用自定义 block reduce + host 求和。
- **待做**：`cub::DeviceReduce::Sum` 等全局 reduce（CUDA Toolkit 自带 CUB）。

---

## 三、可选优化与扩展

### 7. Shared memory 缓存 PI

- **现状**：已部分完成。`monte_carlo_kernel` 已引入 shared-memory tiling 缓存 AND 描述符（fanin/complement）。
- **待做**：进一步评估 PI 级缓存和 warp 协作生成随机 PI 的收益。

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

- **现状**：`main_axsim.cpp` 已支持命令行（路径/pattern/seed/normalizer/位序）；`manual_metrics_test.cpp` 可直接做手算结果对齐。
- **待做**：补自动化单元测试/CI。

### 13. 文档与 README

- **现状**：README 已增加 **Current status**、Make/CUDA 说明、MRED/δ 提示；**TODO** 本文档已同步进度。
- **待做**：ABC/AIG 流程落地后，补充“从文件运行”与命令行示例；MEM 或多值 metrics 若实现则更新表格。

---

## 汇总表

| 序号 | 项 | 状态 | 说明 |
|------|--------------|------|------|
| 1 | ref 来源 | **已完成** | `run_gpu_metrics_pair` 已落地 |
| 2 | 大规模 pattern | **已完成** | 多轮 launch 已支持 |
| 3 | ABC/AIG 对接 | **部分完成** | 文件读取流程可用，第三方 Verilog 兼容性仍需增强 |
| 4 | 输出 gather kernel | 待做 | 替代多次 `cudaMemcpy2D` |
| 5 | CUDA 错误检查 | **已完成** | 关键 cuda API 与 kernel launch 已覆盖 |
| 6 | CUB reduction | 待做 | device 全局 reduce |
| 7 | Shared memory PI | **部分完成** | AND 描述符 shared-memory tiling 已完成 |
| 8 | curand | 可选 | |
| 9 | PI coalesce | 可选 | |
| 10 | MEM | 可选 | |
| 11 | 多值 MRED/MSE | 可选 | |
| 12 | 示例与测试 | **部分完成** | CLI + 手算测试已完成；自动化 CI 待补 |
| 13 | 文档 | **部分完成** | README/TODO 已更新，后续按新特性继续同步 |

完成 **2、3** 后，库即可在**真实 AIG 文件**流程中完整使用；**1** 的库内封装为体验优化；**4、5、6** 提升健壮性与性能；其余按需求选做。
