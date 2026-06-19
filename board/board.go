package board

import "go-ai-infer/inference"

const (
	Size   = inference.BoardSize
	Points = inference.BoardPoints

	Empty = 0
	Black = 1
	White = 2
)

type board struct {
	round int

	// boards[0] 当前局面
	// boards[1] 前 1 手局面
	// boards[2] 前 2 手局面
	// boards[3] 前 3 手局面
	boards [4][Points]int

	// passes[0] 最后一手是否 pass
	// passes[1] 倒数第二手是否 pass
	passes [2]bool

	// blackHashes：轮到黑走时，历史上出现过的局面 hash
	// whiteHashes：轮到白走时，历史上出现过的局面 hash
	// 初始空盘面不插入，pass 产生的局面也不插入
	blackHashes map[uint64]bool
	whiteHashes map[uint64]bool

	// 当前局面 hash。合法性判断中实际会用 analyzeBoard 重新得到 hash，
	// 这里主要用于 move 后维护当前状态。
	hash uint64
}

// 对外暴露一个 Board 别名，方便其他 package 使用。
// 小写 board 和小写方法仍然满足你给出的函数签名。
type Board = board

func New() *Board {
	return &board{
		blackHashes: make(map[uint64]bool),
		whiteHashes: make(map[uint64]bool),
	}
}

// 可选：给其他 package 用的大写 wrapper。
func (cur *board) Round() int                     { return cur.round }
func (cur *board) Mask() [Points]int              { return cur.mask() }
func (cur *board) Move(x int, y int) int          { return cur.move(x, y) }
func (cur *board) Tensor() inference.Features     { return cur.tensor() }
func (cur *board) Score() [3]int                  { return cur.score() }
func (cur *board) IsFinish() int                  { return cur.is_finish() }
func (cur *board) IsEyes(x int, y int) int        { return cur.is_eyes(x, y) }

type group struct {
	color      int
	stones     []int
	libertyCnt int
	hash       uint64
}

type positionAnalysis struct {
	// gid[p] 表示点 p 所属棋块编号。
	// 空点为 -1。
	gid [Points]int

	groups []group

	// 当前整盘棋的 Zobrist hash
	hash uint64
}

type legalResult struct {
	newHash       uint64
	captured      [4]int
	capturedCount int
}

var zobrist = initZobrist()

func initZobrist() [Points][3]uint64 {
	var z [Points][3]uint64
	var x uint64 = 0x9e3779b97f4a7c15

	for i := 0; i < Points; i++ {
		for c := 1; c <= 2; c++ {
			x += 0x9e3779b97f4a7c15
			z[i][c] = splitmix64(x)
		}
	}
	return z
}

func splitmix64(x uint64) uint64 {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

func currentColor(round int) int {
	if round%2 == 0 {
		return Black
	}
	return White
}

func opponent(color int) int {
	if color == Black {
		return White
	}
	return Black
}

func inRange(x int, y int) bool {
	return 0 <= x && x < Size && 0 <= y && y < Size
}

func idx(x int, y int) int {
	return x*Size + y
}

func neighbors(p int, out *[4]int) int {
	cnt := 0
	x := p / Size
	y := p % Size

	if x > 0 {
		out[cnt] = p - Size
		cnt++
	}
	if x+1 < Size {
		out[cnt] = p + Size
		cnt++
	}
	if y > 0 {
		out[cnt] = p - 1
		cnt++
	}
	if y+1 < Size {
		out[cnt] = p + 1
		cnt++
	}
	return cnt
}

// analyzeBoard 一次性分析当前盘面的所有棋块、气数和棋块 hash。
// 时间复杂度 O(N)，空间复杂度 O(N)。
func analyzeBoard(b [Points]int) positionAnalysis {
	var a positionAnalysis
	for i := 0; i < Points; i++ {
		a.gid[i] = -1
	}

	var libertyMark [Points]int
	var queue [Points]int
	var ns [4]int

	for p := 0; p < Points; p++ {
		if b[p] == Empty || a.gid[p] != -1 {
			continue
		}

		gid := len(a.groups)
		mark := gid + 1
		color := b[p]
		g := group{color: color}

		head, tail := 0, 0
		queue[tail] = p
		tail++
		a.gid[p] = gid

		for head < tail {
			q := queue[head]
			head++

			g.stones = append(g.stones, q)
			g.hash ^= zobrist[q][color]

			ncnt := neighbors(q, &ns)
			for i := 0; i < ncnt; i++ {
				n := ns[i]
				s := b[n]

				if s == Empty {
					if libertyMark[n] != mark {
						libertyMark[n] = mark
						g.libertyCnt++
					}
				} else if s == color && a.gid[n] == -1 {
					a.gid[n] = gid
					queue[tail] = n
					tail++
				}
			}
		}

		a.hash ^= g.hash
		a.groups = append(a.groups, g)
	}

	return a
}

func hasSmall(arr *[4]int, cnt int, x int) bool {
	for i := 0; i < cnt; i++ {
		if arr[i] == x {
			return true
		}
	}
	return false
}

func (cur *board) ensureMaps() {
	if cur.blackHashes == nil {
		cur.blackHashes = make(map[uint64]bool)
	}
	if cur.whiteHashes == nil {
		cur.whiteHashes = make(map[uint64]bool)
	}
}

// legalAtWithAnalysis 判断 color 能不能下在 p。
// 要点：
// 1. 只检查 p 周围最多 4 个邻居；
// 2. 提子通过相邻对方棋块 libertyCnt == 1 判断；
// 3. 自杀通过“有直接空邻居 / 连接己方多气棋块 / 能提子”判断；
// 4. 新 hash 通过棋块 hash 增量得到，不复制棋盘，不模拟 move。
// 单点复杂度 O(1)。
func (cur *board) legalAtWithAnalysis(
	p int,
	color int,
	a *positionAnalysis,
	checkRepeat bool,
) (bool, legalResult) {
	var res legalResult

	if p < 0 || p >= Points || cur.boards[0][p] != Empty {
		return false, res
	}

	opp := opponent(color)
	selfOK := false
	captureOK := false

	var ns [4]int
	ncnt := neighbors(p, &ns)

	for i := 0; i < ncnt; i++ {
		n := ns[i]
		stone := cur.boards[0][n]

		if stone == Empty {
			selfOK = true
			continue
		}

		gid := a.gid[n]
		if gid < 0 {
			continue
		}

		g := &a.groups[gid]

		if stone == color {
			if g.libertyCnt > 1 {
				selfOK = true
			}
		} else if stone == opp {
			if g.libertyCnt == 1 && !hasSmall(&res.captured, res.capturedCount, gid) {
				captureOK = true
				res.captured[res.capturedCount] = gid
				res.capturedCount++
			}
		}
	}

	if !captureOK && !selfOK {
		return false, res
	}

	newHash := a.hash ^ zobrist[p][color]
	for i := 0; i < res.capturedCount; i++ {
		gid := res.captured[i]
		newHash ^= a.groups[gid].hash
	}
	res.newHash = newHash

	if checkRepeat {
		next := opponent(color)

		if next == Black {
			if cur.blackHashes[newHash] {
				return false, res
			}
		} else {
			if cur.whiteHashes[newHash] {
				return false, res
			}
		}
	}

	return true, res
}

// mask 返回当前行动方所有合法落点，不包含 pass。
// 时间复杂度 O(N)：一次 analyzeBoard O(N)，然后每个点 O(1) 判断。
func (cur *board) mask() [Points]int {
	var ans [Points]int

	if cur.is_finish() == 1 {
		return ans
	}

	a := analyzeBoard(cur.boards[0])
	color := currentColor(cur.round)

	for p := 0; p < Points; p++ {
		if cur.boards[0][p] != Empty {
			continue
		}

		ok, _ := cur.legalAtWithAnalysis(p, color, &a, true)
		if ok {
			ans[p] = 1
		}
	}

	return ans
}

func (cur *board) pushBoard(next [Points]int, isPass bool) {
	cur.boards[3] = cur.boards[2]
	cur.boards[2] = cur.boards[1]
	cur.boards[1] = cur.boards[0]
	cur.boards[0] = next

	cur.passes[1] = cur.passes[0]
	cur.passes[0] = isPass
}

// move 执行落子或 pass。
// 合法返回 0，不合法返回 1 且 board 不变。
// 时间复杂度 O(N)：
// pass 是 O(N)，因为需要移动四张 [361]int 盘面；
// 普通落子是 analyze O(N) + 提子最多 O(N) + 盘面移动 O(N)。
func (cur *board) move(x int, y int) int {
	if cur.is_finish() == 1 {
		return 1
	}

	cur.ensureMaps()

	if x == -1 && y == -1 {
		cur.pushBoard(cur.boards[0], true)
		cur.round++
		return 0
	}

	if !inRange(x, y) {
		return 1
	}

	p := idx(x, y)

	// 快速过滤：目标点非空直接返回
	if cur.boards[0][p] != Empty {
		return 1
	}

	color := currentColor(cur.round)
	a := analyzeBoard(cur.boards[0])

	ok, res := cur.legalAtWithAnalysis(p, color, &a, true)
	if !ok {
		return 1
	}

	nextBoard := cur.boards[0]
	nextBoard[p] = color

	for i := 0; i < res.capturedCount; i++ {
		gid := res.captured[i]
		for _, s := range a.groups[gid].stones {
			nextBoard[s] = Empty
		}
	}

	cur.pushBoard(nextBoard, false)
	cur.round++
	cur.hash = res.newHash

	nextColor := currentColor(cur.round)
	if nextColor == Black {
		cur.blackHashes[cur.hash] = true
	} else {
		cur.whiteHashes[cur.hash] = true
	}

	return 0
}

// tensor 转换成 inference.Features。
// 通道：
// 0 当前黑，1 当前白，2 前 1 手黑，3 前 1 手白，
// 4 前 2 手黑，5 前 2 手白，6 前 3 手黑，7 前 3 手白，
// 8 当前行动方，黑走为全 1，白走为全 0。
// 时间复杂度 O(N)。
func (cur *board) tensor() inference.Features {
	var f inference.Features

	for k := 0; k < 4; k++ {
		b := &cur.boards[k]
		blackPlane := 2 * k
		whitePlane := 2*k + 1

		for p := 0; p < Points; p++ {
			s := b[p]

			if s == Black {
				f[blackPlane*Points+p] = 1
			} else if s == White {
				f[whitePlane*Points+p] = 1
			}
		}
	}

	if currentColor(cur.round) == Black {
		base := 8 * Points
		for p := 0; p < Points; p++ {
			f[base+p] = 1
		}
	}

	return f
}

// score 数目数，返回 [黑, 白, 中立]。
// 时间复杂度 O(N)：每个空点最多入队一次，每个棋子直接计数一次。
func (cur *board) score() [3]int {
	var ans [3]int

	b := &cur.boards[0]
	var visited [Points]bool
	var queue [Points]int
	var ns [4]int

	for p := 0; p < Points; p++ {
		if b[p] == Black {
			ans[0]++
			continue
		}

		if b[p] == White {
			ans[1]++
			continue
		}

		if visited[p] {
			continue
		}

		head, tail := 0, 0
		queue[tail] = p
		tail++
		visited[p] = true

		cnt := 0
		hasBlack := false
		hasWhite := false

		for head < tail {
			q := queue[head]
			head++
			cnt++

			ncnt := neighbors(q, &ns)
			for i := 0; i < ncnt; i++ {
				n := ns[i]

				if b[n] == Empty {
					if !visited[n] {
						visited[n] = true
						queue[tail] = n
						tail++
					}
				} else if b[n] == Black {
					hasBlack = true
				} else if b[n] == White {
					hasWhite = true
				}
			}
		}

		if hasBlack && !hasWhite {
			ans[0] += cnt
		} else if hasWhite && !hasBlack {
			ans[1] += cnt
		} else {
			ans[2] += cnt
		}
	}

	return ans
}

// is_finish 判断是否连续两手 pass。
// 时间复杂度 O(1)。
func (cur *board) is_finish() int {
	if cur.passes[0] && cur.passes[1] {
		return 1
	}
	return 0
}

// is_eyes 判断当前点是否是当前行动方的眼。
// 定义：对方不能合法下进这个空位，且该空位周围存在己方棋子，才认为是眼。
// 时间复杂度 O(N)：一次 analyzeBoard O(N)，单点合法性判断 O(1)。
func (cur *board) is_eyes(x int, y int) int {
	if !inRange(x, y) {
		return 0
	}

	p := idx(x, y)
	if cur.boards[0][p] != Empty {
		return 0
	}

	// 检查周围是否存在己方棋子，避免把对方地盘误判为己方眼
	color := currentColor(cur.round)
	var ns [4]int
	ncnt := neighbors(p, &ns)
	hasOwn := false
	for i := 0; i < ncnt; i++ {
		if cur.boards[0][ns[i]] == color {
			hasOwn = true
			break
		}
	}
	if !hasOwn {
		return 0
	}

	a := analyzeBoard(cur.boards[0])
	opp := opponent(color)

	ok, _ := cur.legalAtWithAnalysis(p, opp, &a, true)
	if ok {
		return 0
	}

	return 1
}