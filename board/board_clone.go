package board

// Clone 返回当前 board 的深拷贝，包含独立的 ko 历史 map。
//
// 为什么必须用它：board 结构体里的 blackHashes / whiteHashes 是 map（引用类型），
// 直接用 `nb := *cur` 只会复制指针，父子局面共享同一张 map。MCTS 在副本上 Move 时，
// move() 会执行 cur.blackHashes[hash] = true，从而：
//  1. 把假想局面写进调用方的真实棋盘，搜索后真实对局的 superko 判断被污染；
//  2. 不同搜索分支之间打劫历史互相串扰。
// Clone 单独复制两张 map，使每个 MCTS 节点拥有完全独立的历史，与 Python copy() 一致。
//
// 本方法纯属新增，不改变任何已有逻辑。
func (cur *board) Clone() *board {
	nb := *cur // round / boards / passes / hash 都是值类型，会被真复制

	nb.blackHashes = make(map[uint64]bool, len(cur.blackHashes))
	for k := range cur.blackHashes {
		nb.blackHashes[k] = true
	}

	nb.whiteHashes = make(map[uint64]bool, len(cur.whiteHashes))
	for k := range cur.whiteHashes {
		nb.whiteHashes[k] = true
	}

	return &nb
}
