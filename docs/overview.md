# 项目整体概况

## 模型结构

使用 ResNet 处理 19x19 围棋局面。

**输入：**

```text
batch_size * 9 * 19 * 19
```

9 个通道：前 8 个通道表示最近 4 个历史局面，每个局面用黑/白两个通道；最后 1 个通道表示当前先手方，用全 1 或全 0 表示。

**输出**：

模型采用多任务学习结构，共享 ResNet 主干，并包含 policy、value、score、ownership 四个输出头。

```text
policy     : [B, 362]，361 个棋盘点 + pass，对 logits 执行 softmax 后的概率
value      : [B, 1]，当前先手最终赢/输，范围 [-1, 1]
score      : [B, 1]，当前先手最终赢/输多少目
ownership : [B, 2, 19, 19]，每个点输出黑/白两类 logits；label 为 [B, 19, 19]，0=黑，1=白，-1=未知。未知点使用 ignore_index=-1，不参与 ownership loss。
```

## 训练流程

整体流程：

```text
监督学习 → 自博弈 → 用自博弈数据继续训练 CNN
```

### 1. 监督学习

使用人类棋谱训练初始 CNN。一盘棋按每一步拆成多个训练样本。

每个样本：

```text
input      : [9,19,19]，当前局面
policy     : int64，棋手实际落子的类别标签，0-360 表示棋盘点，361 表示 pass
value      : [1]，当前先手视角的终局胜负，1=赢，-1=输
```

监督学习的问题：

1. policy 只记录棋手实际落子的类别标签，但同一局面可能有多个合理落点；
2. 数据集中无 pass 样本，监督学习后被训练成 0；

监督学习阶段只训练 policy/value。

### 2. 自博弈训练

自博弈使用当前 CNN + MCTS 生成棋局。CNN 给 MCTS 提供 policy/value/score/ownership，MCTS 搜索后选择落子，并把搜索结果作为新训练数据。

自博弈阶段联合训练 policy/value/score/ownership。

自博弈样本：

```text
input      : [9,19,19]，当前局面
policy     : [362]，MCTS 访问次数分布，visit_count / total_visit_count
value      : [1]，当前先手视角终局胜负，1=赢，-1=输
score      : [1]，当前先手视角终局目差，正数=赢，负数=输
ownership  : [19,19]，终局点归属，0=黑，1=白，-1=未知；-1 不参与 ownership loss
```

与监督学习阶段不同，这里的 policy 不是 one-hot，而是 MCTS 搜索出来的概率分布，包含 361 个点和 pass。

### 终局处理

数目：棋子点直接归属对应颜色；空点按连通块判断：只接触黑棋则归黑，只接触白棋则归白，否则记为未知。

终局判断器：【暂时待定】好的终局应满足棋盘已经较充分落子，黑白地盘边界基本明确，双方开始选择 pass，从而方便用简化规则数目。

## 工程实现

Python 负责 CNN 训练和推理。Go 可以负责 MCTS 和自博弈并发。

### 自博弈并发架构

自博弈可以同时维护较多 MCTS 实例，例如 100 个，但用 `semaphore` 限制真正同时跑搜索的数量，例如 8~12 个，避免 CPU 被打满。

MCTS 搜索到需要 CNN 评估的 leaf 后，释放 CPU 资源，把局面放入有限长度的推理队列；队列满时阻塞，形成背压，避免请求无限堆积。

推理端由 batcher 统一收集请求，满足 `batch_size` 或 `max_wait_ms` 后，通过 HTTP 调 Python 模型做 batch 推理。

```text
MCTS pool → CPU semaphore → inference queue → batcher → Python CNN
```

这样既能保持 GPU batch 推理效率，也不会让 CPU 或推理队列失控。

## 疑问

终局判断器

如何学出pass
