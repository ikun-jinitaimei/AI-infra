# HPC / Performance Engineering / AI Systems Foundation

**从 Python 与生物信息学，走向底层系统与性能工程。**

本项目以 HelloHPC 2026 为实验载体，记录如何理解 baseline、分析开销、提出优化假设、修改实现、验证正确性，再用 benchmark 判断是否保留改动。代码覆盖 ARM CPU、OpenMP、MPI、Ascend NPU 算子与 LLM Serving；重点是可解释的优化过程，而不只是比赛分数。

An evidence-driven learning project bridging scientific Python and AI systems: CPU vectorization, memory access, parallel execution, NPU kernels, and inference serving.

## 背景与目标

我是上海交通大学医学院硕士研究生，本科与硕士阶段主要背景是基础医学、生物信息学和肿瘤免疫。主要使用 Python，有 Linux、服务器、GPU 和科研计算经验，目前逐步转向 AI / 大模型方向，关注 LLM post-training、Agent systems、AI Infra、推理服务和强化学习。

参加这次校内 HPC 竞赛，是为了补齐 C/C++、编译、CPU 架构、内存层次、SIMD、OpenMP、MPI 和性能分析基础。本项目是这一学习过程的阶段性记录，不代表已经掌握所有系统技术。方案分析、代码实现与实验整理使用了 Codex 辅助，官方框架及第三方组件的来源保留在 [许可说明](THIRD_PARTY_NOTICES.md) 中。

## 建议阅读顺序

1. **[MaiMoe：从解析开销到硬件指令](docs/case-studies.md#maimoe)** — 最完整的多阶段优化链。
2. **[Miniclash：SIMD更宽是否一定更快](docs/case-studies.md#miniclash)** — 固定种子对照与未采用的优化。
3. **[RSM：片上存储与短段固定开销](docs/case-studies.md#rsm)** — NPU组批、缓冲区错误和性能演进。
4. **[学习地图](docs/learning-map.md)** — 已实践内容、尚缺证据与AI Systems的联系。

## 问题、假设与结果

| 项目 | Baseline / 待解决问题 | 优化与预期机制 | 已有结果 |
|---|---|---|---|
| [MaiMoe](cpu/maimoe/) | 接手实现重复解析、字符串构造及哈希工作 | 紧凑事件、复用前缀、ARM SHA、NEON渲染；减少重复运算及分配 | 最大样例3706.430→616.737 ms，约6.01×；四组检查通过 |
| [Miniclash](cpu/miniclash/) | 标量候选搜索；每个候选有相似整数运算 | 多条NEON指令批处理候选；保持筛选语义 | 同轮固定种子微测量：8路19.3977 s→16路15.7171 s，约1.23×；端到端收益另测 |
| [Accelerate](cpu/accelerate/) | 查询×中心的距离、点积、sin/exp计算开销 | 查询分块、NEON双精度、向量数学、范围检查外提、OpenMP | 最终公开large-b三次最慢1178.475 ms；正确性通过 |
| [RSM](npu/ragged-softmax/) | 独立处理短段，DMA、归约和标量交互重复 | 连续段缓存、联合短段、融合softmax、Gather | v23公开综合加速比5.303×；完整公开检查通过 |
| [Blackhole](cpu/blackhole/) | 重复插值模板与权重、场数据复制和通信 | 模板复用、MPI共享窗口、NEON与分阶段Fourier收缩 | 四例完整数值检查通过；huge 26.991 s；没有匹配baseline时间，不补造加速比 |
| [Mahjx](cpu/mahjx/) | 重复枚举、状态计算及概率扫描 | 精确筛选、缓存、OpenMP根分支并行、整数重数 | 已测版两局合计414.530 s；没有匹配baseline时间 |
| [Rune](cpu/rune/) | 调度需兼顾关键路径、资源与setup成本 | 公开模型驱动，按批量分摊setup成本 | 1152个官方本地场景合法，337/337 |
| [LLM Serving](npu/llm-serving/) | 容器缓存/临时目录故障阻止服务启动 | 修复可写HOME与临时目录，按官方流程验证服务 | Stage1 OJ反馈33.411 output token/s、raw-score 30；Stage2未验证 |
| [Cheatsheet](cheatsheet/) | 用有限文本指导Agent生成正确且快速的实现 | 保留CSA思路，补尺寸、计数位宽与空掩码边界 | 397 tokens，格式通过；最终补丁未重跑Agent |

数据来自历史实验，非本次整理重新运行。**本地估分、公开加速比、OJ反馈不能混用。** 逐项证据、测量口径和限制见 [results.md](docs/results.md)；可机读的结果摘录见 [benchmarks](benchmarks/README.md)。

## 有记录的迭代过程

```text
MaiMoe（最大样例；累计改动，并非逐项隔离消融）
接手版3706.430 ms → 解析1343.395 → 硬件SHA1125.801
→ 局部排序/栈上事件880.307 → 缓冲与前缀复用688.367
→ 向量渲染656.027 → 专用渲染621.028 → 最终复验616.737

RSM（相对官方baseline的公开综合加速比）
v17 4.809× → v19 4.946× → v20 5.228× → v21 5.256× → v23 5.303×
```

RSM v18曾因补齐后临时缓冲容量不足失败；v19修复后通过。v24得到用户反馈“正确性通过、5分”，没有完整OJ报告；最后v25仅改工作划分，未编译或设备测试，放在候选目录，不作为已验证成果。

## 我从这些实验中建立的认识

- **性能问题先分层。** 重复解析、内存复制、数学函数、同步和负载不均需要不同处理，不能统一归结为“算力不够”。
- **正确性边界影响优化设计。** 归约补齐容量、尾部向量、浮点运算次序和计数位宽都需要检查。
- **微基准不等于应用收益。** Miniclash一次掩码归约微优化约快3.6%，但端到端没有稳定收益，最终未采用。
- **拒绝无收益改动也是结果。** 更大的SIMD批宽、PGO、额外I/O线程和某些缓存分块没有稳定收益时回退。
- **能解释机制不等于已测到机制。** 当前证据以阶段计时和实现对照为主；没有系统的cache-miss、带宽、IPC或NUMA硬件计数器分析，因此相关硬件瓶颈只能作为假设。

## 与 AI Systems 的关系

CPU内存访问与数据复用训练的是分析数据移动成本的能力，可迁移到GPU HBM与片上存储；SIMD和OpenMP帮助理解并行粒度与调度开销；MPI帮助理解进程间通信与同步。但 **NEON不等于CUDA，MPI不等于NCCL**：本项目尚未完成CUDA/Triton算子开发、NCCL多机通信或post-training/RL训练系统实验。

直接与AI工作负载相关的实践是 Ascend C 算子和 vLLM Ascend 服务。其余题目保持原有HPC语境，不强行包装成大模型项目。

## 目录与复现

```text
cpu/                       选出的CPU实现；大型框架只保留相对上游补丁
npu/
  ragged-softmax/
    verified-v23/          本地完整验证版本
    candidate-v25/         未验证最后候选
  llm-serving/
    stage1/                已验证服务配置
    stage2-candidate/      仅静态审查的八卡配置
cheatsheet/                算法指导及提交配置
benchmarks/                去除环境标识后的指标摘录、CSV及来源摘要
docs/                      优化案例、结果口径、学习地图、复现及目录审计
```

从 [复现步骤](docs/reproduction.md) 获取固定版本官方框架，再覆盖实现或应用补丁。ARM NEON、Ascend内核与LLM服务需要匹配的硬件和依赖；仓库不包含模型、容器镜像、比赛输入数据、隐藏评测或账号配置。

## 来源、边界与后续工作

上游为 [HPC-SJTU/hellohpc-2nd](https://github.com/HPC-SJTU/hellohpc-2nd)，参考commit为 `d7e85db2c403528479c47771c0ab65198a07c18e`。保留CC BY-NC-SA 4.0许可文本及第三方独立许可，例如Miniclash的MIT许可。

接下来值得补充的是受控的硬件计数器测量、线程扩展曲线、单一变量消融和独立复现脚本。它们列在学习计划中，不作为已经完成的内容。当前仓库的价值是保存一个有成功、失败与证据限制的性能工程闭环。

## 按案例阅读与检查

各题目录现在都有 README，按“问题与起点 → 假设与修改 → 验证与限制 → 学习重点”组织。完整阶段表与同轮/跨轮比较边界见 [优化演进](docs/optimization-history.md)，MaiMoe 的阶段数据另提供 [CSV](benchmarks/progression.csv)。

可以先执行 `python tools/check_repository.py` 检查导出源码摘要、相对文档链接和意外的大型产物。这只验证仓库文件一致性，不运行 ARM/NPU 性能测试。更详细的后续实验方向见 [学习路线](docs/learning-roadmap.md)。
