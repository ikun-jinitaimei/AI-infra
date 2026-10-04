# LLM Serving：从可运行到可测量

## 问题与起点

在官方容器与硬件环境运行 vLLM Ascend 服务，并接受准确性/吞吐评测。

## 假设与修改

Stage1 修复容器 HOME 与临时目录不可写等运行问题；Stage2 对实际镜像配置进行静态审查，整理并行、量化与执行选项。

## 验证与限制

Stage1 用户提供 OJ 反馈 accepted，33.411021341667414 output token/s；本地另一轮 32.10558663000127，与 OJ 分开。Stage2 未完成模型启动、精度或性能验证。

## 代码与复现

本目录保留：stage1/、stage2-candidate/。官方基线与接口需按[复现说明](../../docs/reproduction.md)获取。详细版本演进见[实验记录](../../docs/optimization-history.md)，数值与证据见[结果说明](../../docs/results.md)。

## 学习重点

部署问题与性能问题要区分；配置与文档相符不代表模型已经成功运行或获得加速。
