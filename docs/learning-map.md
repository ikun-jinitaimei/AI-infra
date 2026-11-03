# 已实践内容与下一步学习

| 基础主题 | 仓库中实际实践 | 证据或局限 | 与AI Systems的联系 |
|---|---|---|---|
| Linux / HPC | Slurm资源申请、日志定位、容器可写目录与设备环境 | 原始账户配置不公开；不是集群运维项目 | 隔离运行环境、资源申请、故障排查 |
| C/C++编译 | Make/CMake、Python内嵌C编译、OpenMP编译、局部编译参数实验 | 尚无编译器pass或IR优化分析 | 理解扩展模块、kernel编译与ABI |
| CPU架构 | AArch64 NEON、ARM SHA、目标CPU参数 | 不以此声称掌握全部微架构 | 理解硬件能力与可移植性 |
| Cache / memory | 查询分块、紧凑事件、数据复用、减少复制 | 未提供系统硬件计数器验证 | HBM流量与片上复用的分析思路 |
| 单核 / SIMD | Miniclash固定种子批宽对照、NEON计算 | 并行批任务需另外测量 | 并行粒度、寄存器压力的概念 |
| OpenMP | Accelerate查询并行、Mahjx根分支枚举 | 缺完整线程数扩展与NUMA消融 | CPU工作线程调度、并行开销 |
| MPI | Blackhole单节点共享窗口与归约 | 不包含多节点强/弱扩展测试 | 通信、同步及数据分布；不是NCCL实践 |
| Profiling | 阶段计时、源码开销分析、内核/端到端对照 | 缺perf/PMU/设备timeline完整记录 | 测量先于调参 |
| NPU kernel | Ascend C组批、DMA、UB复用、归约 | v23有公开验证；v25未测 | GPU kernel可借鉴的存储与调度问题 |
| LLM inference | Apptainer和vLLM Ascend服务，Stage1测量 | Stage2未运行；未使用SGLang | 服务启动、吞吐、TTFT/TPOT、并发约束 |

## 不包装成已完成经验的内容

CUDA/Triton独立算子开发、NCCL多机通信、LLM post-training训练流水线、RL训练系统、生产级Agent运行时，都仍是后续方向。Cheatsheet是向优化Agent提供算法说明，不等于实现Agent系统。

## 后续实验计划（未执行）

1. 选择一个CPU案例，在固定机器、绑核及相同输入下测量cycles、instructions、cache misses、分支与带宽；先验证平台支持的计数器事件。
2. 为OpenMP案例生成1/2/4/8等线程扩展曲线，区分调度、带宽、同步与NUMA影响。
3. 对组批/数据布局/编译参数作单变量消融，保留所有重复测量，不只记最快一次。
4. 选一个简单归约或softmax，在CUDA/Triton上重新实现并profile；比较机制，不混用CPU/NPU/GPU时间。
5. 在可用资源下完成Stage2启动、精度、SLA与并发测量，再决定配置，而非把文档推荐参数称作最优。

面试时可重点讨论：一次收益最大的优化、一次错误修复、一次回退，以及一项当前证据不能回答的问题。这里展示的是可追溯的学习过程，不是技术关键词清单。
