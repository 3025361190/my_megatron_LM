# my_megatron_LM

## 1. 项目简介

本项目实现了一套**简易的 Megatron 风格分布式训练框架**，主要使用了：

- `Tensor Parallelism (TP)`
- `Data Parallelism (DP)`

项目的主要目的是通过自己动手实现一个可运行、可阅读、可修改的最小版本，来加深对 Megatron 关键技术细节的理解。

## 2. 项目结构

本项目主要包含四个部分。

### 2.1 目标模型的单卡参考实现

对应目录：

- [single_model](./single_model/)

这一部分是项目的**单卡参考模型**，也是自行实现的 GPT 风格模型。它的作用是：

- 提供目标模型结构的基础版本
- 帮助理解模型本身的前向过程
- 作为后续分布式版本的结构对照

也就是说，`single_model` 更像是“目标模型的原型”，用于回答：

**如果不考虑分布式，这个模型本身应该长什么样。**

### 2.2 Megatron 框架下的模型结构

对应目录中的核心文件：

- [megatron/model.py](./megatron/model.py)

这一部分是在 Megatron 风格框架下，对模型并行结构的实现。它包含了分布式训练里最关键的一些并行模块，例如：

- 行切线性层 `RowParallelLinear`
- 列切线性层 `ColumnParallelLinear`
- 词表并行 `VocabParallelEmbedding`
- 并行交叉熵 `vocab_parallel_cross_entropy`
- 并行 Attention
- 并行 MLP
- GPT 主体结构

### 2.3 分布式初始化：通信组与随机状态管理

对应文件：

- [megatron/distributed.py](./megatron/distributed.py)
- [megatron/rng.py](./megatron/rng.py)

这一部分负责分布式训练的基础设施，主要包括三块内容。

第一块是**通信组初始化**。

项目中维护了：

- 全局通信组
- Tensor Parallel 组
- Data Parallel 组
- Model Parallel 相关组

通过这些通信组，项目实现了 TP + DP 训练所需要的基本通信拓扑。

第二块是**通信原语实现**。

除了通信组划分之外，[megatron/distributed.py](./megatron/distributed.py) 中还实现了张量并行训练中会用到的一些基础通信原语，例如：

- 输入复制
- 张量切分
- 张量聚合
- 张量归约

这些通信原语是行切线性层、列切线性层、并行 Embedding 和并行交叉熵能够工作的基础。  

第三块是**随机状态管理**。

项目除了维护默认随机状态，还专门维护了 TP 专用随机状态，用来支持：

- TP 权重初始化
- TP 相关 dropout 或其他随机操作

### 2.4 训练代码

对应文件：

- [megatron/train.py](./megatron/train.py)

这一部分负责把前面的各个模块真正串起来，形成一条完整的训练流程。主要包括：

- 分布式环境初始化
- TP / DP 组初始化
- 随机种子初始化
- 模型实例化
- DDP 包装
- 数据加载
- 优化器构造
- 梯度累计
- 最小训练循环
