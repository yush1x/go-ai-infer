# Runner

`runner` 负责并发执行多盘 SelfPlay，并把正常结束的棋局提交给 Python 保存服务。

## 使用

```go
cfg := runner.DefaultConfig()
cfg.Games = 100
cfg.Concurrency = 8
cfg.ValueMCTSWeight = 0.2

storage, err := runner.NewHTTPStorageClient(
    runner.DefaultStorageURL,
    runner.DefaultStorageTimeout,
)
if err != nil {
    return err
}

r, err := runner.New(searcher, storage, cfg)
if err != nil {
    return err
}

stats, err := r.Run(ctx)
```

其中 `searcher` 通常是启用 `SelfPlay=true` 的 `*mcts.Searcher`，其下层共享同一个 `inference.Batcher`。

`ValueMCTSWeight` 控制训练 value 标签中 MCTS 根节点估值的占比，范围为 `[0,1]`；默认 `0` 保持纯终局标签。

## 行为

- 最多同时运行 `Concurrency` 盘棋。
- 每盘棋由 `selfplay.Play` 从空棋盘开始。
- 正常结束的棋局按 `docs/selfplay_storage_protocol.md` 提交给 Python。
- 保存请求串行执行，每盘只提交一次。
- 保存失败会记录日志并丢弃，不自动重试。
- 超过 400 手、搜索失败、非法动作和取消均不会提交训练数据。

`Stats` 返回生成、保存、失败和样本数量等统计信息。Runner 不负责创建或关闭 Inference Batcher，其生命周期由 Main 管理。
