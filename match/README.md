# Match

`match` 使用两个不同的 MCTS Searcher 并发运行多盘对局，自动交替黑白，统计双方胜局和胜率，并把全部棋局写入一个 JSON 文件。

运行入口：

```bash
go run ./matchgame
```

主要参数集中在 `matchgame/main.go` 文件顶部。Python 推理服务需要提供：

```text
POST /predict/a
POST /predict/b
```

两个模型分别使用独立的 `HTTPClient + Batcher`，避免不同模型的局面进入同一个推理 batch。

运行期间，交互终端会持续原地刷新全部棋局的状态和当前手数，并分别显示两个模型的推理 batch 大小、满批率和延迟。日志重定向到文件时改为每隔一段时间输出汇总，同时在每盘结束时输出胜负信息。

运行完成后，打开 `matchgame/viewer.html`，选择生成的 JSON 文件即可浏览全部棋局。JSON 为紧凑单文件格式，每一步保存 Go 实际执行后的完整棋盘，浏览器不需要重新实现提子、劫或合法性规则。

