# go-ai-infer

使用 Go 实现围棋自博弈数据生成流程。Python 负责神经网络训练和推理，Go 负责棋盘规则、MCTS、自博弈并发以及训练数据生成。

## 整体结构

项目分为 5 个主要模块：

### 1. Board

负责维护围棋棋局状态并实现棋盘相关功能：

- 保存当前棋盘和历史局面
- 执行落子、提子和 pass
- 判断劫、自杀以及合法落点
- 返回当前所有合法动作
- 判断棋局是否结束
- 计算目数、胜负和终局 ownership
- 生成神经网络推理需要的 `[9, 19, 19]` 输入张量
- 提供棋盘复制能力，供 MCTS 模拟落子

### 2. Inference

负责调用 Python 神经网络推理服务：

- 接收 MCTS 提交的局面评估请求
- 使用有限队列提供背压
- 请求达到 `batch_size` 或等待达到 `max_wait` 后组成 batch
- 通过 HTTP 二进制协议调用 Python 推理服务
- 返回 policy、value、score 和 ownership
- 处理超时、取消、服务错误以及模块关闭

### 3. MCTS

负责使用蒙特卡洛树搜索选择动作：

- 使用 Board 判断和执行合法动作
- 完成节点选择、扩展和结果回传
- 调用 Inference 评估叶节点
- 屏蔽非法动作并重新归一化 policy
- 统计根节点各动作的访问次数
- 返回访问次数分布和最终选择的动作

### 4. SelfPlay

负责生成一盘完整的自博弈棋局和训练数据：

- 维护一盘棋使用的 Board
- 每一步调用 MCTS 选择动作
- 记录每一步的输入张量、当前行动方和 MCTS 访问次数分布
- 终局后为所有步骤补充 value、score 和 ownership 标签
- 整理并保存训练样本，例如交给 Python 写入 HDF5 文件

### 5. Runner / Main

负责启动和管理整个自博弈任务：

- 创建并共享 Inference batcher
- 并发运行多个 SelfPlay
- 控制棋局数量和 CPU 并发度
- 管理配置、日志、错误、取消和优雅退出
- 汇总生成棋局和训练样本的统计信息

## 模块依赖

```text
Runner / Main
      |
      v
  SelfPlay --------> Board
      |
      v
    MCTS ----------> Board
      |
      v
  Inference -------> Python CNN
```

主要调用流程：

```text
Runner 启动多个 SelfPlay
    -> SelfPlay 使用 MCTS 搜索下一步
    -> MCTS 使用 Board 模拟棋局
    -> MCTS 将叶节点提交给 Inference
    -> Inference 合并请求并调用 Python CNN
    -> MCTS 根据推理结果完成搜索
    -> SelfPlay 执行动作并记录训练样本
    -> 棋局结束后补充终局标签并保存数据
```

## 相关文档

- [项目模型与训练流程](docs/overview.md)
- [Python 推理服务二进制协议](docs/inference_protocol.md)
