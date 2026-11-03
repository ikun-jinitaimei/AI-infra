# 复现与环境

本仓库提供优化产物，不复制官方评测器或隐藏测试。需要在对应硬件环境获得官方依赖、测试工具和数据。历史参赛队列已经撤除的记录不代表现在有可用资源。

## 1. 获取匹配的官方框架

```bash
git clone https://github.com/HPC-SJTU/hellohpc-2nd.git
git -C hellohpc-2nd checkout d7e85db2c403528479c47771c0ab65198a07c18e
```

## 2. 选择一个实现覆盖对应题目

| 本仓库 | 官方目录 | 操作 |
|---|---|---|
| cpu/accelerate | 03-accelerate | 复制src/solver.py与env.sh |
| cheatsheet | 04-cheatsheet | 按官方提交布局使用SKILL.md和submission.yaml |
| cpu/miniclash/source_code | 05-miniclash/source_code | 覆盖同名源文件，保留第三方LICENSE.txt |
| cpu/rune/policy.py | 06-rune | 按官方policy接口放入提交文件 |
| cpu/maimoe/solution | 07-maimoe/solution | 覆盖Kernel.cpp和KstroParam.toml |
| cpu/blackhole/optimization.patch | 08-blackhole | 在官方仓库根目录使用git apply --directory=08-blackhole |
| cpu/mahjx/optimization.patch | 09-mahjx | 同上，补丁对应已测版本 |
| npu/ragged-softmax/verified-v23/solution | 10-kernel/solution | 优先选择已测v23，候选v25需单独目录验证 |
| npu/llm-serving/stage1 | 11-llm | 按官方Stage0/Stage1流程使用build.def与start.sh |

示例（从匹配的官方仓库根目录执行，并替换补丁绝对路径；Mahjx将08-blackhole换成09-mahjx并使用对应补丁）：

```bash
git apply --check --directory=08-blackhole /path/to/hellohpc-performance-lab/cpu/blackhole/optimization.patch
git apply --directory=08-blackhole /path/to/hellohpc-performance-lab/cpu/blackhole/optimization.patch
```

## 3. 按各题官方README运行

CPU优化主要面向ARM64 Kunpeng环境，NEON及ARM SHA指令路径不能直接视为x86通用实现。NPU内核面向Ascend 910B3及比赛CANN环境；LLM需要匹配驱动、vLLM Ascend镜像、模型及官方注入的HELLOHPC环境变量。模型权重与容器镜像不在仓库中。

先通过正确性检查，再按官方预热、重复次数和计时范围测量性能。不要把不同设备、线程数、精度或输入规模的成绩直接比较。Stage2和RSM v25需从编译/启动与正确性验证开始，不应跳过检查直接宣称更快。
