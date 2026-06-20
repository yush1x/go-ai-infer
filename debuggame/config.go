package main

const (
	// Python 模型推理接口。
	defaultPredictURL = "http://127.0.0.1:8000/predict"

	// 调试棋局 JSON 输出路径。可以修改目录和文件名。
	defaultOutputFile = "debuggame/game2.json"

	// 单局最多执行多少步；达到后仍未连续两次 pass，则以 max_moves 结束。
	defaultMaxMoves = 500

	// 每一步 MCTS 搜索执行的模拟次数。
	defaultSimulations = 100

	// MCTS 在利用和探索之间的平衡系数。
	defaultCPuct = float32(1.5)

	// 自博弈根节点 Dirichlet 噪声参数。
	defaultDirichletAlpha = float32(0.03)
	defaultDirichletEps   = float32(0.05)

	// 过滤非法动作和添加噪声后、重新归一化前的 pass policy 下限。
	// 0 表示关闭；建议从 0.01~0.05 开始测试。
	defaultPassPolicyFloor = float32(0.00)

	// 热门非 pass 走法均为己方眼时，加到 pass PUCT 分数上的 bonus。
	// 0 表示关闭；建议从 0.05~0.5 开始，超过 1 通常很强。
	defaultPassBonus = float32(0.8)
)
