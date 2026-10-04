# Benchmark evidence

本目录保存历史报告中选出的指标字段，没有重新执行计算。原始报告SHA256与摘录来源见 [evidence-index.json](evidence-index.json)。摘录删除账户、远程路径和输入内容；部分报告只保留汇总数值，不足以重建完整测量分布。

- `evidence/accelerate-final.json`：3个新进程样本、取最大值、正确性和计时范围。
- `evidence/maimoe-final.json`：各组均值、单位及重复测量样本。
- `evidence/blackhole-final.json`、`mahjx-final.json`：完整公开检查汇总。
- `evidence/rune-*.json`：官方场景与扰动验证的汇总，移除种子与输入配置。
- `evidence/rsm-v*.json`：公开综合加速比与运行状态。CLI分数字段不等于正式排行榜分数。v16的已知容量缺陷见优化案例，不能以当次通过代替所有边界验证。
- `rsm-progression.csv`：从上述JSON提取的版本曲线。

LLM Stage1正式结果来自用户提供的OJ截图，目前没有可发布的机器可读原始报告；不伪造JSON“原始证据”。Miniclash阶段时间取自实验日志整理，详见优化案例。

## 新实验应记录什么

硬件、编译器和依赖版本、源码commit/hash、线程数与亲和性、输入规模、预热次数、重复次数、全部样本、统计方式、计时范围及正确性检查。比较时保持这些条件一致；跨平台比较需明确环境差异。

没有匹配baseline、只有一次测量或没有profiler证据时，保留这个限制，不补造数字和瓶颈结论。
