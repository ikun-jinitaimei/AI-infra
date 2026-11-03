# Ragged Softmax Moments：变长短段与片上复用

## 问题与起点

参考实现按变长分段计算 softmax、均值与中心矩，需要满足精度约束。

## 假设与修改

多段输入批量 DMA、片上复用、短段列 lane 联合运算、融合归约及向量搬移。保留中心矩与残差修正；v18 暴露补齐后临时容量不足，修复后继续验证。

## 验证与限制

verified-v23 公开综合加速比 5.30305414320573，完整检查通过；candidate-v25 未编译、未设备测试。v16 虽曾通过但后来发现相关容量缺陷，不推荐复用。

## 代码与复现

本目录保留：verified-v23/solution、candidate-v25/solution。官方基线与接口需按[复现说明](../../docs/reproduction.md)获取。详细版本演进见[实验记录](../../docs/optimization-history.md)，数值与证据见[结果说明](../../docs/results.md)。

## 学习重点

变长负载的固定开销、片上存储容量和精度约束必须共同设计；减少 DMA 次数并不自动保证更快。
