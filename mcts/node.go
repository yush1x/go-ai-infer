package mcts

import (
	"go-ai-infer/board"
)

// 动作编码：0..360 为棋盘点（p = x*Size + y），361 = pass。
const (
	PassAction = board.Points // 361
	PolicySize = PassAction + 1 // 362，与 inference.PolicySize 一致
)

// Node 是 MCTS 树节点，对应 Python 版 Node。
// board 用指针持有，因为 board 内部含 map，必须配合 Clone 做深拷贝。
type Node struct {
	board    *board.Board  // 该节点对应的局面（独立深拷贝）
	prior    float32       // CNN policy（已 mask + 归一化）
	children map[int]*Node // action -> 子节点
	visits   int32         // N
	valueSum float32       // W，从该节点行动方视角累积
}

func newNode(b *board.Board, prior float32) *Node {
	return &Node{board: b, prior: prior}
}

// q 返回节点平均价值；未访问时为 0，对应 Python Node.q。
func (n *Node) q() float32 {
	if n.visits == 0 {
		return 0
	}
	return n.valueSum / float32(n.visits)
}

// actionToXY 把动作编号转成棋盘坐标（仅非 pass 动作有效）。
func actionToXY(action int) (x, y int) {
	return action / board.Size, action % board.Size
}
