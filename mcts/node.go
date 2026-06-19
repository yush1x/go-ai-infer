package mcts

import (
	"errors"
	"fmt"

	"go-ai-infer/board"
)

// 动作编码：0..360 为棋盘点（p = x*Size + y），361 = pass。
const (
	PassAction = board.Points   // 361
	PolicySize = PassAction + 1 // 362，与 inference.PolicySize 一致
)

// Node 是 MCTS 树节点，对应 Python 版 Node。
// 非根节点的 board 延迟到该节点首次被选中时才创建，避免展开时为所有候选动作复制棋盘。
type Node struct {
	parent   *Node         // 根节点为 nil
	action   int           // 从 parent 到当前节点的动作；根节点为 -1
	board    *board.Board  // 当前局面；未被选中的非根节点为 nil
	prior    float32       // CNN policy（已 mask + 归一化）
	children map[int]*Node // action -> 子节点
	visits   int32         // N
	valueSum float32       // W，从该节点行动方视角累积
}

func newRootNode(b *board.Board) *Node {
	return &Node{action: -1, board: b}
}

func newChildNode(parent *Node, action int, prior float32) *Node {
	return &Node{
		parent: parent,
		action: action,
		prior:  prior,
	}
}

// ensureBoard 在节点首次被选中时生成动作执行后的局面，之后复用该局面。
func (n *Node) ensureBoard() error {
	if n.board != nil {
		return nil
	}
	if n.parent == nil || n.parent.board == nil {
		return errors.New("mcts: cannot create node board without parent board")
	}

	b := n.parent.board.Clone()
	var result int
	if n.action == PassAction {
		result = b.Move(-1, -1)
	} else {
		x, y := actionToXY(n.action)
		result = b.Move(x, y)
	}
	if result != 0 {
		return fmt.Errorf("mcts: child action %d is illegal", n.action)
	}

	n.board = b
	return nil
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
