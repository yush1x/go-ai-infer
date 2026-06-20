# SelfPlay

`selfplay` 负责从空棋盘开始完成一盘自博弈，并返回可供训练的数据。它不负责并发调度、日志聚合、Python 通信或 HDF5 写入；这些职责留给后续的 Runner 和 Writer。

## 输入

```go
result := selfplay.Play(ctx, searcher)
```

- `ctx`：用于取消和超时。
- `searcher`：通常传入开启 `SelfPlay` 模式的 `*mcts.Searcher`。
- 棋局固定从 `board.New()` 开始，不接收外部局面。

## 输出

正常终局时，`Result.Game.Samples` 包含每一步的训练样本：

```text
Features  float32 [9,19,19]  落子前的局面
Policy    float32 [362]       MCTS 根节点访问次数分布
Value     float32             当前行动方视角的终局胜负
Score     float32             当前行动方视角的终局目差
Ownership int8 [19,19]        0=黑，1=白，-1=中立/未知
```

`Features` 和 `Policy` 在每次落子前记录。棋局正常结束后，再根据最终结果为所有步骤回填 `Value`、`Score` 和 `Ownership`。

## 终局与失败

- Board 以连续两次 pass 判定正常终局。
- 单盘默认最多执行 `400` 手，也可通过 `PlayConfig.MaxMoves` 配置。
- 达到最大手数仍未结束时，按当前局面强制结算并返回 `StatusMaxMoves`，训练数据仍可保存。
- 搜索失败、非法动作和取消也会返回独立状态、已执行手数、最后动作及错误，供 Main 或未来 Runner 输出日志。

## 职责边界

```text
SelfPlay：生成一盘棋和训练样本
Runner：并发运行多盘、统计状态、记录日志
Writer：把成功样本交给 Python 并写入 HDF5
```
