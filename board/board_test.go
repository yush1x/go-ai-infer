package board

import (
	"testing"

	"go-ai-infer/inference"
)

// ============================================================================
// 辅助函数
// ============================================================================

// newBoardWithStones 创建一个新 board 并直接放置棋子。
// stones 格式: map[idx]color
func newBoardWithStones(stones map[int]int) *board {
	b := &board{
		blackHashes: make(map[uint64]bool),
		whiteHashes: make(map[uint64]bool),
	}
	for p, c := range stones {
		b.boards[0][p] = c
	}
	return b
}

// ============================================================================
// 1. 基础工具函数测试
// ============================================================================

func TestCurrentColor(t *testing.T) {
	tests := []struct {
		round int
		want  int
	}{
		{0, Black}, {1, White}, {2, Black}, {3, White},
		{100, Black}, {101, White},
	}
	for _, tt := range tests {
		got := currentColor(tt.round)
		if got != tt.want {
			t.Errorf("currentColor(%d) = %d, want %d", tt.round, got, tt.want)
		}
	}
}

func TestOpponent(t *testing.T) {
	if opponent(Black) != White {
		t.Error("opponent(Black) should be White")
	}
	if opponent(White) != Black {
		t.Error("opponent(White) should be Black")
	}
}

func TestInRange(t *testing.T) {
	// 角点
	if !inRange(0, 0) {
		t.Error("(0,0) should be in range")
	}
	if !inRange(18, 18) {
		t.Error("(18,18) should be in range")
	}

	// 界外
	if inRange(-1, 0) {
		t.Error("(-1,0) should be out of range")
	}
	if inRange(0, -1) {
		t.Error("(0,-1) should be out of range")
	}
	if inRange(19, 0) {
		t.Error("(19,0) should be out of range")
	}
	if inRange(0, 19) {
		t.Error("(0,19) should be out of range")
	}
}

func TestIdx(t *testing.T) {
	// idx(0,0) == 0
	if idx(0, 0) != 0 {
		t.Error("idx(0,0) should be 0")
	}
	// idx(18,18) == 360
	if idx(18, 18) != 360 {
		t.Errorf("idx(18,18) = %d, want 360", idx(18, 18))
	}
	// 往返: p → (x,y) → p
	for p := 0; p < Points; p++ {
		x, y := p/Size, p%Size
		if idx(x, y) != p {
			t.Errorf("idx(%d,%d) = %d, want %d", x, y, idx(x, y), p)
		}
	}
}

func TestNeighbors(t *testing.T) {
	var ns [4]int

	t.Run("corner", func(t *testing.T) {
		cnt := neighbors(0, &ns)
		if cnt != 2 {
			t.Errorf("corner (0,0) should have 2 neighbors, got %d", cnt)
		}
		foundRight, foundDown := false, false
		for i := 0; i < cnt; i++ {
			if ns[i] == 1 {
				foundRight = true
			}
			if ns[i] == Size {
				foundDown = true
			}
		}
		if !foundRight || !foundDown {
			t.Errorf("corner (0,0) neighbors wrong: %v", ns[:cnt])
		}
	})

	t.Run("edge", func(t *testing.T) {
		p := idx(0, 5)
		cnt := neighbors(p, &ns)
		if cnt != 3 {
			t.Errorf("edge (0,5) should have 3 neighbors, got %d", cnt)
		}
	})

	t.Run("center", func(t *testing.T) {
		p := idx(9, 9)
		cnt := neighbors(p, &ns)
		if cnt != 4 {
			t.Errorf("center (9,9) should have 4 neighbors, got %d", cnt)
		}
	})
}

func TestHasSmall(t *testing.T) {
	arr := [4]int{10, 20, 30, 0}

	if !hasSmall(&arr, 3, 20) {
		t.Error("hasSmall should find 20 in first 3")
	}
	if hasSmall(&arr, 3, 0) {
		t.Error("hasSmall should NOT find 0 in first 3 (0 is at index 3)")
	}
	if hasSmall(&arr, 3, 99) {
		t.Error("hasSmall should NOT find 99")
	}
}

func TestSplitmix64(t *testing.T) {
	a := splitmix64(0x9e3779b97f4a7c15)
	b := splitmix64(0x9e3779b97f4a7c15)
	if a != b {
		t.Error("splitmix64 should be deterministic")
	}
	if a == 0 {
		t.Error("splitmix64 should produce non-zero output")
	}
}

// ============================================================================
// 2. analyzeBoard 测试
// ============================================================================

func TestAnalyzeBoard_Empty(t *testing.T) {
	var b [Points]int
	a := analyzeBoard(b)

	if len(a.groups) != 0 {
		t.Errorf("empty board should have 0 groups, got %d", len(a.groups))
	}
	if a.hash != 0 {
		t.Errorf("empty board hash should be 0, got %d", a.hash)
	}
	for p := 0; p < Points; p++ {
		if a.gid[p] != -1 {
			t.Errorf("empty board gid[%d] = %d, want -1", p, a.gid[p])
		}
	}
}

func TestAnalyzeBoard_SingleStone(t *testing.T) {
	var b [Points]int
	p := idx(9, 9)
	b[p] = Black

	a := analyzeBoard(b)

	if len(a.groups) != 1 {
		t.Fatalf("should have 1 group, got %d", len(a.groups))
	}
	g := a.groups[0]
	if g.color != Black {
		t.Error("group color should be Black")
	}
	if len(g.stones) != 1 {
		t.Errorf("group should have 1 stone, got %d", len(g.stones))
	}
	if g.stones[0] != p {
		t.Errorf("group stone should be at %d, got %d", p, g.stones[0])
	}
	if g.libertyCnt != 4 {
		t.Errorf("center stone should have 4 liberties, got %d", g.libertyCnt)
	}
	if a.gid[p] != 0 {
		t.Errorf("gid of stone should be 0, got %d", a.gid[p])
	}
}

func TestAnalyzeBoard_CornerStone(t *testing.T) {
	var b [Points]int
	p := idx(0, 0)
	b[p] = Black

	a := analyzeBoard(b)
	if len(a.groups) != 1 {
		t.Fatalf("should have 1 group, got %d", len(a.groups))
	}
	if a.groups[0].libertyCnt != 2 {
		t.Errorf("corner stone should have 2 liberties, got %d", a.groups[0].libertyCnt)
	}
}

func TestAnalyzeBoard_ConnectedGroup(t *testing.T) {
	var b [Points]int
	p1 := idx(9, 9)
	p2 := idx(9, 10)
	b[p1] = Black
	b[p2] = Black

	a := analyzeBoard(b)

	if len(a.groups) != 1 {
		t.Fatalf("connected stones should form 1 group, got %d", len(a.groups))
	}
	g := a.groups[0]
	if len(g.stones) != 2 {
		t.Errorf("group should have 2 stones, got %d", len(g.stones))
	}
	if g.libertyCnt != 6 {
		t.Errorf("connected group should have 6 liberties, got %d", g.libertyCnt)
	}
	if a.gid[p1] != a.gid[p2] {
		t.Error("connected stones should share same gid")
	}
}

func TestAnalyzeBoard_SeparateGroups(t *testing.T) {
	var b [Points]int
	pb := idx(3, 3)
	pw := idx(15, 15)
	b[pb] = Black
	b[pw] = White

	a := analyzeBoard(b)

	if len(a.groups) != 2 {
		t.Fatalf("should have 2 groups, got %d", len(a.groups))
	}
	colors := map[int]bool{}
	for _, g := range a.groups {
		colors[g.color] = true
	}
	if !colors[Black] || !colors[White] {
		t.Error("should have one Black and one White group")
	}
}

func TestAnalyzeBoard_HashConsistency(t *testing.T) {
	var b1 [Points]int
	b1[idx(3, 3)] = Black
	b1[idx(3, 4)] = White

	var b2 [Points]int
	b2[idx(3, 3)] = Black
	b2[idx(3, 4)] = White

	if analyzeBoard(b1).hash != analyzeBoard(b2).hash {
		t.Error("identical boards should have identical hashes")
	}
}

// ============================================================================
// 3. legalAtWithAnalysis 测试
// ============================================================================

func TestLegalAtWithAnalysis_EmptyPoint(t *testing.T) {
	b := newBoardWithStones(nil)
	a := analyzeBoard(b.boards[0])

	ok, _ := b.legalAtWithAnalysis(idx(9, 9), Black, &a, false)
	if !ok {
		t.Error("placing on empty point should be legal")
	}
}

func TestLegalAtWithAnalysis_OccupiedPoint(t *testing.T) {
	stones := map[int]int{idx(9, 9): Black}
	b := newBoardWithStones(stones)
	a := analyzeBoard(b.boards[0])

	ok, _ := b.legalAtWithAnalysis(idx(9, 9), Black, &a, false)
	if ok {
		t.Error("placing on occupied point should be illegal")
	}
}

func TestLegalAtWithAnalysis_OutOfBounds(t *testing.T) {
	b := newBoardWithStones(nil)
	a := analyzeBoard(b.boards[0])

	ok, _ := b.legalAtWithAnalysis(-1, Black, &a, false)
	if ok {
		t.Error("negative index should be illegal")
	}
	ok, _ = b.legalAtWithAnalysis(Points, Black, &a, false)
	if ok {
		t.Error("index >= Points should be illegal")
	}
}

func TestLegalAtWithAnalysis_Capture(t *testing.T) {
	// 构造: B at (1,0), B at (0,1), B at (1,2), W at (1,1)
	// W 的气仅剩 (2,1) → 黑下 (2,1) 可提
	stones := map[int]int{
		idx(1, 0): Black,
		idx(0, 1): Black,
		idx(1, 1): White,
		idx(1, 2): Black,
	}
	b := newBoardWithStones(stones)
	a := analyzeBoard(b.boards[0])

	// 验证白子只有 1 气
	for i := range a.groups {
		if a.groups[i].color == White && a.groups[i].libertyCnt != 1 {
			t.Errorf("White group should have 1 liberty, got %d", a.groups[i].libertyCnt)
		}
	}

	// 黑下 (2,1) 提白
	p := idx(2, 1)
	ok, res := b.legalAtWithAnalysis(p, Black, &a, false)
	if !ok {
		t.Error("capture move should be legal")
	}
	if res.capturedCount != 1 {
		t.Errorf("should capture 1 group, got %d", res.capturedCount)
	}
}

func TestLegalAtWithAnalysis_Suicide(t *testing.T) {
	// 空点被敌子包围且无提子 → 自杀
	stones := map[int]int{
		idx(0, 0): Black, idx(0, 1): Black, idx(0, 2): Black,
		idx(1, 0): Black, idx(1, 2): Black,
		idx(2, 0): Black, idx(2, 1): Black, idx(2, 2): Black,
	}
	b := newBoardWithStones(stones)
	a := analyzeBoard(b.boards[0])

	p := idx(1, 1)
	ok, _ := b.legalAtWithAnalysis(p, White, &a, false)
	if ok {
		t.Error("suicide move should be illegal")
	}
}

func TestLegalAtWithAnalysis_Ko(t *testing.T) {
	// 构造 ko 局面
	stones := map[int]int{
		idx(1, 0): Black,
		idx(0, 1): Black,
		idx(1, 1): White,
		idx(1, 2): Black,
	}
	b := newBoardWithStones(stones)
	b.round = 0

	a := analyzeBoard(b.boards[0])
	_, res := b.legalAtWithAnalysis(idx(2, 1), Black, &a, false)

	// 提子后的 hash 放入 whiteHashes（模拟该局面曾轮到白走）
	b.whiteHashes[res.newHash] = true

	// 黑尝试再次在 (2,1) 落子，应被 ko 拒绝
	ok, _ := b.legalAtWithAnalysis(idx(2, 1), Black, &a, true)
	if ok {
		t.Error("ko: repeating a previous position should be illegal")
	}
}

// ============================================================================
// 4. move 测试
// ============================================================================

func TestMove_Pass(t *testing.T) {
	b := New()
	ret := b.Move(-1, -1)
	if ret != 0 {
		t.Error("pass should return 0")
	}
	if b.Round() != 1 {
		t.Errorf("after pass, round should be 1, got %d", b.Round())
	}
	if !b.passes[0] {
		t.Error("passes[0] should be true")
	}
}

func TestMove_LegalMove(t *testing.T) {
	b := New()
	ret := b.Move(9, 9)
	if ret != 0 {
		t.Error("legal move should return 0")
	}
	if b.Round() != 1 {
		t.Errorf("after move, round should be 1, got %d", b.Round())
	}
	if b.boards[0][idx(9, 9)] != Black {
		t.Error("stone should be placed at (9,9)")
	}
}

func TestMove_OutOfBounds(t *testing.T) {
	b := New()
	if b.Move(-2, 0) != 1 {
		t.Error("x out of bounds should return 1")
	}
	if b.Move(0, -2) != 1 {
		t.Error("y out of bounds should return 1")
	}
	if b.Move(19, 0) != 1 {
		t.Error("x >= Size should return 1")
	}
	if b.Move(0, 19) != 1 {
		t.Error("y >= Size should return 1")
	}
}

func TestMove_OccupiedPoint(t *testing.T) {
	b := New()
	b.Move(3, 3) // Black
	ret := b.Move(3, 3)
	if ret != 1 {
		t.Error("placing on occupied point should return 1")
	}
}

func TestMove_CaptureSequence(t *testing.T) {
	b := New()

	// 构造提子局面:
	// Black: (1,0), (0,1), (1,2)
	// White: (1,1) 只有 1 气 at (2,1)
	// Black captures at (2,1)
	// 需要偶数手让黑棋在第6手执行提子:
	// B, W, B, W, B, W(凑手), B(提)
	moves := []struct{ x, y int }{
		{1, 0}, // B round0→1
		{1, 1}, // W round1→2
		{0, 1}, // B round2→3
		{2, 2}, // W round3→4 (随便下)
		{1, 2}, // B round4→5, W(1,1)仅1气 at (2,1)
		{3, 3}, // W round5→6 (凑手，让黑走下一步)
	}

	for _, m := range moves {
		ret := b.Move(m.x, m.y)
		if ret != 0 {
			t.Fatalf("unexpected illegal move at (%d,%d) round=%d color=%d",
				m.x, m.y, b.Round(), currentColor(b.Round()))
		}
	}

	// 白子应在 (1,1)，仅 1 气
	if b.boards[0][idx(1, 1)] != White {
		t.Error("White stone should be at (1,1)")
	}

	// 黑下 (2,1) 提白 (round=6, Black)
	ret := b.Move(2, 1)
	if ret != 0 {
		t.Fatalf("capture move should be legal, got %d (round=%d, color=%d)",
			ret, b.Round(), currentColor(b.Round()))
	}
	if b.boards[0][idx(1, 1)] != Empty {
		t.Error("captured White stone should be removed from (1,1)")
	}
}

func TestMove_GameEndDenial(t *testing.T) {
	b := New()
	b.Move(-1, -1) // pass 1
	b.Move(-1, -1) // pass 2 → game over

	ret := b.Move(3, 3)
	if ret != 1 {
		t.Error("move after game end should return 1")
	}
}

func TestMove_HashRecorded(t *testing.T) {
	b := New()
	b.Move(3, 3) // Black moves
	// 落子后 hash 应该被记录在 whiteHashes（下一手轮到白）
	if len(b.whiteHashes) != 1 {
		t.Errorf("after Black moves, whiteHashes should have 1 entry, got %d", len(b.whiteHashes))
	}
	b.Move(15, 15) // White moves
	// 落子后 hash 应该被记录在 blackHashes（下一手轮到黑）
	if len(b.blackHashes) != 1 {
		t.Errorf("after White moves, blackHashes should have 1 entry, got %d", len(b.blackHashes))
	}
}

// ============================================================================
// 5. score 测试
// ============================================================================

func TestScore_EmptyBoard(t *testing.T) {
	b := New()
	s := b.Score()
	if s[0] != 0 || s[1] != 0 {
		t.Errorf("empty board score should be [0,0,*], got [%d,%d,%d]", s[0], s[1], s[2])
	}
	if s[2] != Points {
		t.Errorf("empty board neutral should be %d, got %d", Points, s[2])
	}
}

func TestScore_AfterMoves(t *testing.T) {
	// 黑下 1 子，白下 1 子 → 各 1 子，剩余空域中立（同时邻黑白）
	b := New()
	b.Move(3, 3)   // B
	b.Move(15, 15) // W

	s := b.Score()
	// 两子相距远，空域同时邻黑邻白，归中立
	if s[0] < 1 {
		t.Errorf("Black should have at least 1 stone, got %d", s[0])
	}
	if s[1] < 1 {
		t.Errorf("White should have at least 1 stone, got %d", s[1])
	}
}

func TestScore_AfterCapture(t *testing.T) {
	b := New()
	// 同 TestMove_CaptureSequence，含提子
	b.Move(1, 0) // B
	b.Move(1, 1) // W
	b.Move(0, 1) // B
	b.Move(2, 2) // W
	b.Move(1, 2) // B
	b.Move(3, 3) // W (凑手)
	b.Move(2, 1) // B captures W at (1,1)

	s := b.Score()
	// 总和始终 361
	if s[0]+s[1]+s[2] != Points {
		t.Errorf("total score should be %d, got %d", Points, s[0]+s[1]+s[2])
	}
	// 被提的白子不应计数
	if b.boards[0][idx(1, 1)] != Empty {
		t.Error("captured stone should be removed")
	}
	// 黑至少 4 子
	if s[0] < 4 {
		t.Errorf("Black should have >=4 stones, got %d", s[0])
	}
}

func TestScore_TerritoryOwnership(t *testing.T) {
	// 黑子围住 (1,1)，同时在远处放白子确保空域归属正确
	stones := map[int]int{
		// 黑子 3x3 方框围住 (1,1)
		idx(0, 0): Black, idx(0, 1): Black, idx(0, 2): Black,
		idx(1, 0): Black, idx(1, 2): Black,
		idx(2, 0): Black, idx(2, 1): Black, idx(2, 2): Black,
		// 白子在远处
		idx(10, 10): White,
	}
	// 让黑棋块有外气避免死棋
	stones[idx(0, 3)] = Black
	stones[idx(1, 3)] = Black
	stones[idx(2, 3)] = Black

	b := newBoardWithStones(stones)

	s := b.Score()
	// 黑子 11 颗（8+3 外气），围住 (1,1) → 至少 12
	// (1,1) 被黑子完全包围，只邻黑 → 归黑
	if s[0] < 12 {
		t.Errorf("Black should have >=12 (11 stones + territory), got %d", s[0])
	}
	// 总和 361
	if s[0]+s[1]+s[2] != Points {
		t.Errorf("total should be %d, got %d", Points, s[0]+s[1]+s[2])
	}
}

func TestScore_TotalAlways361(t *testing.T) {
	b := New()
	// 随机下几手
	for i := 0; i < 10; i++ {
		x, y := (i*7)%19, (i*11)%19
		b.Move(x, y)
	}
	s := b.Score()
	if s[0]+s[1]+s[2] != Points {
		t.Errorf("total score should always be %d, got %d", Points, s[0]+s[1]+s[2])
	}
}

// ============================================================================
// 6. is_finish 测试
// ============================================================================

func TestIsFinish_Initial(t *testing.T) {
	b := New()
	if b.IsFinish() != 0 {
		t.Error("new board should not be finished")
	}
}

func TestIsFinish_OnePass(t *testing.T) {
	b := New()
	b.Move(-1, -1)
	if b.IsFinish() != 0 {
		t.Error("after one pass, game should not be finished")
	}
}

func TestIsFinish_TwoPasses(t *testing.T) {
	b := New()
	b.Move(-1, -1)
	b.Move(-1, -1)
	if b.IsFinish() != 1 {
		t.Error("after two consecutive passes, game should be finished")
	}
}

func TestIsFinish_PassMovePass(t *testing.T) {
	b := New()
	b.Move(-1, -1) // pass
	b.Move(3, 3)   // move → breaks pass chain
	if b.IsFinish() != 0 {
		t.Error("after pass then move, game should not be finished")
	}
}

// ============================================================================
// 7. is_eyes 测试（含 bug 修复验证）
// ============================================================================

func TestIsEyes_RealEye(t *testing.T) {
	// 黑子围住 (1,1)，且棋块在外围有气
	stones := map[int]int{
		idx(0, 0): Black, idx(0, 1): Black, idx(0, 2): Black,
		idx(1, 0): Black, idx(1, 2): Black,
		idx(2, 0): Black, idx(2, 1): Black, idx(2, 2): Black,
		idx(0, 3): Black, idx(1, 3): Black, idx(2, 3): Black,
	}
	b := newBoardWithStones(stones)
	b.round = 0 // Black to move

	if b.IsEyes(1, 1) != 1 {
		t.Error("(1,1) should be Black's eye")
	}
}

func TestIsEyes_NotEye_EmptyBoard(t *testing.T) {
	b := New()
	if b.IsEyes(9, 9) != 0 {
		t.Error("on empty board, any point should NOT be an eye")
	}
}

func TestIsEyes_NotEye_OpponentTerritory(t *testing.T) {
	// Bug 修复验证：
	// 空点完全被白子包围，当前轮到黑走。
	// 白虽不能下该点（自杀），但绝非黑棋的眼！
	stones := map[int]int{
		idx(0, 0): White, idx(0, 1): White, idx(0, 2): White,
		idx(1, 0): White, idx(1, 2): White,
		idx(2, 0): White, idx(2, 1): White, idx(2, 2): White,
		idx(0, 3): White, idx(1, 3): White, idx(2, 3): White,
	}
	b := newBoardWithStones(stones)
	b.round = 0 // Black to move（当前行棋方是黑）

	result := b.IsEyes(1, 1)
	if result != 0 {
		t.Errorf("BUG REGRESSION: (1,1) surrounded by White should NOT be Black's eye, got %d", result)
	}
}

func TestIsEyes_OutOfBounds(t *testing.T) {
	b := New()
	if b.IsEyes(-1, 0) != 0 {
		t.Error("out of bounds should return 0")
	}
	if b.IsEyes(19, 0) != 0 {
		t.Error("out of bounds should return 0")
	}
}

func TestIsEyes_NotEmpty(t *testing.T) {
	b := New()
	b.Move(3, 3)
	if b.IsEyes(3, 3) != 0 {
		t.Error("non-empty point should not be an eye")
	}
}

// ============================================================================
// 8. tensor 测试
// ============================================================================

func TestTensor_Dimensions(t *testing.T) {
	b := New()
	ft := b.Tensor()
	if len(ft) != inference.FeatureSize {
		t.Errorf("tensor length should be %d, got %d", inference.FeatureSize, len(ft))
	}
}

func TestTensor_CurrentPlayerPlane_Black(t *testing.T) {
	b := New()
	ft := b.Tensor()
	base := 8 * Points
	for p := 0; p < Points; p++ {
		if ft[base+p] != 1.0 {
			t.Errorf("current player plane (Black) should be all 1, got %f at %d", ft[base+p], p)
			break
		}
	}
}

func TestTensor_CurrentPlayerPlane_White(t *testing.T) {
	b := New()
	b.Move(3, 3) // Black moves → round=1, White's turn
	ft := b.Tensor()
	base := 8 * Points
	for p := 0; p < Points; p++ {
		if ft[base+p] != 0.0 {
			t.Errorf("current player plane (White) should be all 0, got %f at %d", ft[base+p], p)
			break
		}
	}
}

func TestTensor_StoneEncoding(t *testing.T) {
	b := New()
	b.Move(3, 3) // Black at (3,3)
	ft := b.Tensor()
	p := idx(3, 3)

	if ft[0*Points+p] != 1.0 {
		t.Error("plane 0 (current black) should have 1 at (3,3)")
	}
	if ft[1*Points+p] != 0.0 {
		t.Error("plane 1 (current white) should have 0 at (3,3)")
	}
}

func TestTensor_HistoryPlanes(t *testing.T) {
	b := New()
	b.Move(3, 3)   // B → round 1
	b.Move(15, 15) // W → round 2
	b.Move(4, 4)   // B → round 3

	ft := b.Tensor()

	// boards[0] (current, round=3): B(3,3), B(4,4), W(15,15)
	// boards[1] (prev, round=2):   B(3,3), W(15,15)
	// boards[2] (prev-1, round=1): B(3,3)

	// Plane 0 (current Black): B(3,3), B(4,4) 两子
	pB1 := idx(3, 3)
	pB2 := idx(4, 4)
	if ft[0*Points+pB1] != 1.0 {
		t.Error("plane 0: current Black at (3,3) should be 1")
	}
	if ft[0*Points+pB2] != 1.0 {
		t.Error("plane 0: current Black at (4,4) should be 1")
	}

	// Plane 1 (current White): W(15,15)
	pW := idx(15, 15)
	if ft[1*Points+pW] != 1.0 {
		t.Error("plane 1: current White at (15,15) should be 1")
	}

	// Plane 2 (boards[1] Black): only B(3,3), B(4,4) was just placed
	if ft[2*Points+pB1] != 1.0 {
		t.Error("plane 2: previous Black at (3,3) should be 1")
	}
	if ft[2*Points+pB2] != 0.0 {
		t.Error("plane 2: (4,4) not in boards[1], should be 0")
	}
}

// ============================================================================
// 9. mask 测试
// ============================================================================

func TestMask_EmptyBoard(t *testing.T) {
	b := New()
	m := b.Mask()

	count := 0
	for p := 0; p < Points; p++ {
		if m[p] == 1 {
			count++
		}
	}
	if count != Points {
		t.Errorf("empty board mask should have all %d points legal, got %d", Points, count)
	}
}

func TestMask_AfterGameEnd(t *testing.T) {
	b := New()
	b.Move(-1, -1)
	b.Move(-1, -1)

	m := b.Mask()
	for p := 0; p < Points; p++ {
		if m[p] != 0 {
			t.Error("after game end, mask should be all 0")
			break
		}
	}
}

func TestMask_OccupiedPointsNotInMask(t *testing.T) {
	b := New()
	b.Move(3, 3)

	m := b.Mask()
	if m[idx(3, 3)] != 0 {
		t.Error("occupied point should not be in mask")
	}
}

// ============================================================================
// 10. pushBoard 测试
// ============================================================================

func TestPushBoard_HistoryShift(t *testing.T) {
	b := New()

	b.boards[0][idx(0, 0)] = Black
	b.boards[1][idx(1, 1)] = White
	b.boards[2][idx(2, 2)] = Black
	b.boards[3][idx(3, 3)] = White

	var next [Points]int
	next[idx(4, 4)] = Black

	b.pushBoard(next, false)

	if b.boards[0][idx(4, 4)] != Black {
		t.Error("boards[0] should be new board")
	}
	if b.boards[1][idx(0, 0)] != Black {
		t.Error("boards[1] should be old boards[0]")
	}
	if b.boards[2][idx(1, 1)] != White {
		t.Error("boards[2] should be old boards[1]")
	}
	if b.boards[3][idx(2, 2)] != Black {
		t.Error("boards[3] should be old boards[2]")
	}
}

func TestPushBoard_PassShift(t *testing.T) {
	b := New()
	b.passes[0] = true
	b.passes[1] = false

	var next [Points]int
	b.pushBoard(next, true)

	if !b.passes[0] {
		t.Error("passes[0] should be true (new pass)")
	}
	if !b.passes[1] {
		t.Error("passes[1] should be true (old passes[0])")
	}
}

// ============================================================================
// 11. New / ensureMaps 测试
// ============================================================================

func TestNew(t *testing.T) {
	b := New()
	if b == nil {
		t.Fatal("New() returned nil")
	}
	if b.blackHashes == nil {
		t.Error("blackHashes should be initialized")
	}
	if b.whiteHashes == nil {
		t.Error("whiteHashes should be initialized")
	}
	if b.round != 0 {
		t.Errorf("round should be 0, got %d", b.round)
	}
	if b.hash != 0 {
		t.Errorf("initial hash should be 0, got %d", b.hash)
	}
}

func TestEnsureMaps(t *testing.T) {
	b := &board{} // 不通过 New()，maps 为 nil
	b.ensureMaps()
	if b.blackHashes == nil {
		t.Error("ensureMaps should init blackHashes")
	}
	if b.whiteHashes == nil {
		t.Error("ensureMaps should init whiteHashes")
	}
}

// ============================================================================
// 12. Zobrist 哈希测试
// ============================================================================

func TestZobrist_Initialization(t *testing.T) {
	for i := 0; i < Points; i++ {
		if zobrist[i][Black] == 0 {
			t.Errorf("zobrist[%d][Black] should be non-zero", i)
		}
		if zobrist[i][White] == 0 {
			t.Errorf("zobrist[%d][White] should be non-zero", i)
		}
		if zobrist[i][Black] == zobrist[i][White] {
			t.Errorf("zobrist[%d] Black and White should differ", i)
		}
	}
}

// ============================================================================
// 13. 集成测试：完整对局流程
// ============================================================================

func TestIntegration_FullGame(t *testing.T) {
	b := New()

	// 验证初始状态
	if b.Round() != 0 {
		t.Fatal("initial round should be 0")
	}
	if b.IsFinish() != 0 {
		t.Fatal("initial game should not be finished")
	}

	// 黑落子
	if ret := b.Move(3, 3); ret != 0 {
		t.Fatal("Black's first move failed")
	}
	if b.boards[0][idx(3, 3)] != Black {
		t.Fatal("Black stone not placed")
	}

	// 白落子
	if ret := b.Move(15, 15); ret != 0 {
		t.Fatal("White's first move failed")
	}
	if b.boards[0][idx(15, 15)] != White {
		t.Fatal("White stone not placed")
	}

	// 再下几手
	b.Move(3, 4)   // B
	b.Move(15, 14) // W
	b.Move(3, 5)   // B

	if b.Round() != 5 {
		t.Errorf("round should be 5 after 5 moves, got %d", b.Round())
	}

	// Mask 应该有合法落点
	m := b.Mask()
	hasLegal := false
	for p := 0; p < Points; p++ {
		if m[p] == 1 {
			hasLegal = true
			break
		}
	}
	if !hasLegal {
		t.Error("should have legal moves mid-game")
	}

	// Score 总和恒为 361
	s := b.Score()
	if s[0]+s[1]+s[2] != Points {
		t.Errorf("total score should be %d, got %d", Points, s[0]+s[1]+s[2])
	}

	// 连续两次 pass 终局
	b.Move(-1, -1) // pass
	if b.IsFinish() != 0 {
		t.Error("should not finish after 1 pass")
	}
	b.Move(-1, -1) // pass
	if b.IsFinish() != 1 {
		t.Error("should finish after 2 consecutive passes")
	}

	// 终局后无法落子
	if ret := b.Move(6, 6); ret != 1 {
		t.Error("should not allow move after game end")
	}

	// 终局后 Mask 为空
	m2 := b.Mask()
	for p := 0; p < Points; p++ {
		if m2[p] != 0 {
			t.Error("mask should be empty after game end")
			break
		}
	}

	// Tensor 正常工作
	ft := b.Tensor()
	if len(ft) != inference.FeatureSize {
		t.Errorf("tensor size should be %d, got %d", inference.FeatureSize, len(ft))
	}

	t.Logf("Final score: Black=%d White=%d Neutral=%d", s[0], s[1], s[2])
}
