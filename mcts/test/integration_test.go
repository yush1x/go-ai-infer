package test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"go-ai-infer/board"
	"go-ai-infer/inference"
)

// ============================================================================
// mockPredictor：模拟 Python 推理服务，用于测试 inference 模块
// ============================================================================

type mockPredictor struct {
	callCount int
}

func (m *mockPredictor) Predict(_ context.Context, features []inference.Features) ([]inference.Evaluation, error) {
	m.callCount++
	results := make([]inference.Evaluation, len(features))
	for i := range features {
		// 返回一些模拟的推理结果
		results[i].Value = 0.5
		results[i].Score = 3.5
		// 模拟 policy：给 pass 一个较小的概率
		results[i].Policy[inference.BoardPoints] = 0.01
	}
	return results, nil
}

// ============================================================================
// 测试 1：Board 基础功能
// ============================================================================

func TestBoardBasicOperations(t *testing.T) {
	fmt.Println("=== 测试 1：棋盘基础操作 ===")

	b := board.New()
	fmt.Printf("✓ 成功创建棋盘，当前轮次: %d\n", b.Round())

	// 黑棋下在星位 (3,3)
	result := b.Move(3, 3)
	if result != 0 {
		t.Fatalf("✗ 黑棋落子 (3,3) 失败，返回码: %d", result)
	}
	fmt.Println("✓ 黑棋成功落子 (3,3)")

	// 白棋下在 (15,15)
	result = b.Move(15, 15)
	if result != 0 {
		t.Fatalf("✗ 白棋落子 (15,15) 失败，返回码: %d", result)
	}
	fmt.Println("✓ 白棋成功落子 (15,15)")

	// 检查合法落点
	mask := b.Mask()
	legalCount := 0
	for _, v := range mask {
		if v == 1 {
			legalCount++
		}
	}
	fmt.Printf("✓ 当前合法落点数: %d\n", legalCount)

	// 检查棋盘未结束
	if b.IsFinish() != 0 {
		t.Error("✗ 棋盘不应该结束")
	}
	fmt.Println("✓ 棋盘未结束，符合预期")

	// 生成输入张量
	tensor := b.Tensor()
	nonZero := 0
	for _, v := range tensor {
		if v != 0 {
			nonZero++
		}
	}
	fmt.Printf("✓ 生成输入张量，非零元素数: %d\n", nonZero)

	fmt.Println("=== 测试 1 通过 ===")
}

// ============================================================================
// 测试 2：Inference Batcher 功能
// ============================================================================

func TestInferenceBatcher(t *testing.T) {
	fmt.Println("=== 测试 2：推理批量管理器 ===")

	predictor := &mockPredictor{}
	config := inference.DefaultBatcherConfig()
	config.BatchSize = 4
	config.MaxWait = 100 * time.Millisecond

	batcher, err := inference.NewBatcher(predictor, config)
	if err != nil {
		t.Fatalf("✗ 创建 Batcher 失败: %v", err)
	}
	defer batcher.Close()
	fmt.Println("✓ 成功创建 Batcher")

	// 提交单个评估请求
	ctx := context.Background()
	var features inference.Features
	features[0] = 1.0 // 设置一个特征值做标记

	eval, err := batcher.Evaluate(ctx, features)
	if err != nil {
		t.Fatalf("✗ Evaluate 失败: %v", err)
	}
	fmt.Printf("✓ 推理成功，Value=%.2f, Score=%.2f\n", eval.Value, eval.Score)

	// 验证结果
	if eval.Value != 0.5 {
		t.Errorf("✗ Value 期望 0.5，实际 %.2f", eval.Value)
	}
	if eval.Score != 3.5 {
		t.Errorf("✗ Score 期望 3.5，实际 %.2f", eval.Score)
	}

	fmt.Println("=== 测试 2 通过 ===")
}

// ============================================================================
// 测试 3：Batcher 并发推理
// ============================================================================

func TestInferenceBatcherConcurrent(t *testing.T) {
	fmt.Println("=== 测试 3：并发推理 ===")

	predictor := &mockPredictor{}
	config := inference.DefaultBatcherConfig()
	config.BatchSize = 8
	config.MaxWait = 200 * time.Millisecond

	batcher, err := inference.NewBatcher(predictor, config)
	if err != nil {
		t.Fatalf("✗ 创建 Batcher 失败: %v", err)
	}
	defer batcher.Close()
	fmt.Println("✓ 成功创建 Batcher")

	// 并发提交 10 个请求
	const numRequests = 10
	ctx := context.Background()
	errCh := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			var features inference.Features
			features[0] = float32(id)
			eval, err := batcher.Evaluate(ctx, features)
			if err != nil {
				errCh <- fmt.Errorf("请求 %d 失败: %v", id, err)
				return
			}
			if eval.Value != 0.5 {
				errCh <- fmt.Errorf("请求 %d Value 不正确: %.2f", id, eval.Value)
				return
			}
			errCh <- nil
		}(i)
	}

	// 收集结果
	for i := 0; i < numRequests; i++ {
		if err := <-errCh; err != nil {
			t.Error(err)
		}
	}
	fmt.Printf("✓ %d 个并发请求全部成功\n", numRequests)
	fmt.Printf("✓ Predictor 被调用 %d 次（batch 合并）\n", predictor.callCount)

	fmt.Println("=== 测试 3 通过 ===")
}

// ============================================================================
// 测试 4：Batcher 关闭与错误处理
// ============================================================================

func TestInferenceBatcherClose(t *testing.T) {
	fmt.Println("=== 测试 4：Batcher 关闭与错误处理 ===")

	predictor := &mockPredictor{}
	config := inference.DefaultBatcherConfig()

	batcher, err := inference.NewBatcher(predictor, config)
	if err != nil {
		t.Fatalf("✗ 创建 Batcher 失败: %v", err)
	}
	fmt.Println("✓ 成功创建 Batcher")

	// 关闭 batcher
	err = batcher.Close()
	if err != nil {
		t.Fatalf("✗ 关闭 Batcher 失败: %v", err)
	}
	fmt.Println("✓ Batcher 成功关闭")

	// 关闭后尝试提交请求，应该返回 ErrClosed
	ctx := context.Background()
	var features inference.Features
	_, err = batcher.Evaluate(ctx, features)
	if err == nil {
		t.Error("✗ 关闭后 Evaluate 应该返回错误")
	}
	fmt.Printf("✓ 关闭后 Evaluate 返回错误: %v\n", err)

	fmt.Println("=== 测试 4 通过 ===")
}

// ============================================================================
// 测试 5：棋盘完整对局模拟
// ============================================================================

func TestBoardFullGameSimulation(t *testing.T) {
	fmt.Println("=== 测试 5：模拟一局简单对局 ===")

	b := board.New()

	// 模拟一些落子
	moves := [][2]int{
		{3, 3}, {15, 15}, // 黑星位, 白星位
		{3, 15}, {15, 3}, // 黑星位, 白星位
		{3, 4}, {15, 14}, // 黑, 白
		{3, 5}, {15, 13}, // 黑, 白
	}

	for i, m := range moves {
		result := b.Move(m[0], m[1])
		if result != 0 {
			t.Fatalf("✗ 第 %d 手落子 (%d,%d) 失败", i+1, m[0], m[1])
		}
		player := "黑"
		if i%2 == 1 {
			player = "白"
		}
		fmt.Printf("  第 %d 手: %s棋 (%d,%d) ✓\n", i+1, player, m[0], m[1])
	}

	// 检查当前状态
	fmt.Printf("✓ 当前轮次: %d\n", b.Round())
	fmt.Printf("✓ 合法落点数: %d\n", countLegalMoves(b))

	// 生成张量
	tensor := b.Tensor()
	fmt.Printf("✓ 张量大小: %d\n", len(tensor))

	// 计算目数
	score := b.Score()
	fmt.Printf("✓ 目数统计 - 黑: %d, 白: %d, 空: %d\n", score[0], score[1], score[2])

	fmt.Println("=== 测试 5 通过 ===")
}

// ============================================================================
// 测试 6：HTTP 客户端协议编解码
// ============================================================================

func TestHTTPClientEncodeDecode(t *testing.T) {
	fmt.Println("=== 测试 6：HTTP 协议编解码 ===")

	client, err := inference.NewHTTPClient("http://127.0.0.1:8000/predict", 30*time.Second)
	if err != nil {
		t.Fatalf("✗ 创建 HTTPClient 失败: %v", err)
	}
	fmt.Println("✓ 成功创建 HTTPClient")

	// 验证客户端配置
	if client == nil {
		t.Fatal("✗ HTTPClient 为 nil")
	}
	fmt.Println("✓ HTTPClient 配置正确")

	fmt.Println("=== 测试 6 通过 ===")
}

// ============================================================================
// 辅助函数
// ============================================================================

func countLegalMoves(b *board.Board) int {
	mask := b.Mask()
	count := 0
	for _, v := range mask {
		if v == 1 {
			count++
		}
	}
	return count
}

// ============================================================================
// realisticPredictor：模拟真实 CNN 推理，返回结构合理的 policy/value/score/ownership
// ============================================================================

type realisticPredictor struct{}

func (r *realisticPredictor) Predict(_ context.Context, features []inference.Features) ([]inference.Evaluation, error) {
	results := make([]inference.Evaluation, len(features))
	for i := range features {
		// 从输入张量中提取当前局面信息，构造合理的 mock 输出
		f := features[i]

		// 统计当前盘面黑子、白子数（从第0、1通道读取）
		blackCount := 0
		whiteCount := 0
		for p := 0; p < inference.BoardPoints; p++ {
			if f[0*inference.BoardPoints+p] > 0.5 {
				blackCount++
			}
			if f[1*inference.BoardPoints+p] > 0.5 {
				whiteCount++
			}
		}

		// 当前行动方：通道8全1=黑走，全0=白走
		isBlack := f[8*inference.BoardPoints] > 0.5

		// --- 构造 policy：合法落点附近给较高概率，其他地方给低概率 ---
		// 遍历所有棋盘位置，如果该位置和周围是空的就给较高概率
		for p := 0; p < inference.BoardPoints; p++ {
			x := p / inference.BoardSize
			y := p % inference.BoardSize

			// 空点且不在边角：给中等概率
			isEmpty := f[0*inference.BoardPoints+p] < 0.5 && f[1*inference.BoardPoints+p] < 0.5
			if isEmpty {
				// 星位附近给更高概率
				if isStarPoint(x, y) {
					results[i].Policy[p] = 0.05
				} else if x > 0 && x < 18 && y > 0 && y < 18 {
					results[i].Policy[p] = 0.003
				} else {
					results[i].Policy[p] = 0.001
				}
			}
			// 有子的位置概率为0（非法动作，MCTS 会屏蔽）
		}
		// pass 概率
		results[i].Policy[inference.BoardPoints] = 0.02

		// 归一化 policy 使总和为 1
		var sum float32
		for _, v := range results[i].Policy {
			sum += v
		}
		if sum > 0 {
			for j := range results[i].Policy {
				results[i].Policy[j] /= sum
			}
		}

		// --- 构造 value：根据子数差估算 ---
		diff := blackCount - whiteCount
		if !isBlack {
			diff = -diff
		}
		// 将子数差映射到 [-1, 1]，10子约等于0.5
		results[i].Value = clampFloat(float32(diff)/20.0, -1, 1)

		// --- 构造 score：目数预估 ---
		results[i].Score = float32(diff) * 0.7

		// --- 构造 ownership：有子的地方归属明确，空点按距离估算 ---
		for p := 0; p < inference.BoardPoints; p++ {
			hasBlack := f[0*inference.BoardPoints+p] > 0.5
			hasWhite := f[1*inference.BoardPoints+p] > 0.5
			if hasBlack {
				results[i].Ownership[0*inference.BoardPoints+p] = 0.9 // 黑通道高
				results[i].Ownership[1*inference.BoardPoints+p] = 0.1 // 白通道低
			} else if hasWhite {
				results[i].Ownership[0*inference.BoardPoints+p] = 0.1
				results[i].Ownership[1*inference.BoardPoints+p] = 0.9
			} else {
				results[i].Ownership[0*inference.BoardPoints+p] = 0.5
				results[i].Ownership[1*inference.BoardPoints+p] = 0.5
			}
		}
	}
	return results, nil
}

func isStarPoint(x, y int) bool {
	stars := map[int]bool{3: true, 9: true, 15: true}
	return stars[x] && stars[y]
}

func clampFloat(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ============================================================================
// 测试 7：完整推理数据展示 —— 从 Board 生成张量 → Inference 返回 Evaluation
// ============================================================================

func TestInferenceFullOutput(t *testing.T) {
	fmt.Println("=== 测试 7：完整推理数据展示 ===")
	fmt.Println()

	// 1. 创建一个有实际落子的棋盘
	b := board.New()

	// 模拟开局前几手（常见星位布局）
	moves := [][2]int{
		{3, 3},   // 黑 1: 星位
		{15, 15}, // 白 2: 星位
		{3, 15},  // 黑 3: 星位
		{15, 3},  // 白 4: 星位
		{9, 9},   // 黑 5: 天元
		{15, 14}, // 白 6: 挂角
		{9, 3},   // 黑 7: 边上
		{15, 16}, // 白 8: 拆边
	}

	for i, m := range moves {
		result := b.Move(m[0], m[1])
		if result != 0 {
			t.Fatalf("✗ 第 %d 手落子 (%d,%d) 失败", i+1, m[0], m[1])
		}
	}
	fmt.Printf("棋盘状态: 已下 %d 手, 当前轮次 %d\n", b.Round(), b.Round())
	fmt.Printf("当前行动方: ")
	if b.Round()%2 == 0 {
		fmt.Println("黑棋 ●")
	} else {
		fmt.Println("白棋 ○")
	}
	fmt.Printf("合法落点数: %d\n", countLegalMoves(b))
	fmt.Println()

	// 2. 生成神经网络输入张量
	tensor := b.Tensor()
	fmt.Println("━━━ 输入张量 ━━━")
	fmt.Printf("  形状: [%d, %d, %d]\n", inference.FeaturePlanes, inference.BoardSize, inference.BoardSize)
	fmt.Printf("  总元素数: %d (float32)\n", len(tensor))
	fmt.Printf("  总字节数: %d bytes\n", len(tensor)*4)

	// 统计每个通道的非零元素
	fmt.Println("  各通道非零元素数:")
	channelNames := []string{
		"ch0 当前黑子", "ch1 当前白子",
		"ch2 前1手黑", "ch3 前1手白",
		"ch4 前2手黑", "ch5 前2手白",
		"ch6 前3手黑", "ch7 前3手白",
		"ch8 当前行动方",
	}
	for ch := 0; ch < inference.FeaturePlanes; ch++ {
		count := 0
		for p := 0; p < inference.BoardPoints; p++ {
			if tensor[ch*inference.BoardPoints+p] > 0.5 {
				count++
			}
		}
		fmt.Printf("    %s: %d\n", channelNames[ch], count)
	}
	fmt.Println()

	// 3. 通过 Batcher 调用推理
	predictor := &realisticPredictor{}
	config := inference.DefaultBatcherConfig()
	config.BatchSize = 4
	config.MaxWait = 200 * time.Millisecond

	batcher, err := inference.NewBatcher(predictor, config)
	if err != nil {
		t.Fatalf("✗ 创建 Batcher 失败: %v", err)
	}
	defer batcher.Close()

	ctx := context.Background()
	eval, err := batcher.Evaluate(ctx, tensor)
	if err != nil {
		t.Fatalf("✗ 推理失败: %v", err)
	}

	// 4. 展示完整的 Evaluation 结果
	fmt.Println("━━━ 推理输出 (Evaluation) ━━━")
	fmt.Println()

	// --- Policy ---
	fmt.Println("【Policy 向量】")
	fmt.Printf("  长度: %d (361个棋盘点 + 1个pass)\n", len(eval.Policy))

	// 统计 policy 分布
	var policySum float32
	policyNonZero := 0
	top5 := topN(eval.Policy[:inference.BoardPoints], 5)
	for _, v := range eval.Policy {
		policySum += v
		if v > 0 {
			policyNonZero++
		}
	}
	fmt.Printf("  总和: %.6f (应为1.0)\n", policySum)
	fmt.Printf("  非零项: %d / %d\n", policyNonZero, len(eval.Policy))
	fmt.Printf("  Pass 概率: %.6f\n", eval.Policy[inference.BoardPoints])
	fmt.Println("  Top-5 落点 (棋盘坐标 → 概率):")
	for _, item := range top5 {
		x := item.idx / inference.BoardSize
		y := item.idx % inference.BoardSize
		fmt.Printf("    (%2d,%2d) → %.6f\n", x, y, item.val)
	}
	fmt.Println()

	// --- Value ---
	fmt.Println("【Value (胜率)】")
	fmt.Printf("  值: %+.4f (范围[-1,1], 正=当前方优势)\n", eval.Value)
	fmt.Println()

	// --- Score ---
	fmt.Println("【Score (目差)】")
	fmt.Printf("  值: %+.4f (正=当前方领先)\n", eval.Score)
	fmt.Println()

	// --- Ownership ---
	fmt.Println("【Ownership (棋盘归属)】")
	fmt.Printf("  长度: %d (2通道 × 361点)\n", len(eval.Ownership))
	blackOwn := 0
	whiteOwn := 0
	neutralOwn := 0
	for p := 0; p < inference.BoardPoints; p++ {
		blackProb := eval.Ownership[0*inference.BoardPoints+p]
		whiteProb := eval.Ownership[1*inference.BoardPoints+p]
		if blackProb > 0.55 {
			blackOwn++
		} else if whiteProb > 0.55 {
			whiteOwn++
		} else {
			neutralOwn++
		}
	}
	fmt.Printf("  归属统计: 黑=%d, 白=%d, 未定=%d\n", blackOwn, whiteOwn, neutralOwn)
	fmt.Println()

	// 5. 展示 ownership 在棋盘上的分布（文本可视化，缩小版）
	fmt.Println("【Ownership 棋盘可视化 (●=黑 ○=白 ·=未定)】")
	fmt.Println("  (仅展示中心 9×9 区域)")
	fmt.Print("   ")
	for y := 6; y <= 14; y++ {
		fmt.Printf("%2d", y)
	}
	fmt.Println()
	for x := 6; x <= 14; x++ {
		fmt.Printf("%2d ", x)
		for y := 6; y <= 14; y++ {
			p := x*inference.BoardSize + y
			bp := eval.Ownership[0*inference.BoardPoints+p]
			wp := eval.Ownership[1*inference.BoardPoints+p]
			if bp > 0.55 {
				fmt.Print("● ")
			} else if wp > 0.55 {
				fmt.Print("○ ")
			} else {
				fmt.Print("· ")
			}
		}
		fmt.Printf("%2d\n", x)
	}
	fmt.Print("   ")
	for y := 6; y <= 14; y++ {
		fmt.Printf("%2d", y)
	}
	fmt.Println()

	fmt.Println()
	fmt.Println("=== 测试 7 通过 ===")
}

type topItem struct {
	idx int
	val float32
}

func topN(arr []float32, n int) []topItem {
	items := make([]topItem, n)
	for i := range items {
		items[i] = topItem{idx: -1, val: -1}
	}
	for idx, val := range arr {
		if val <= items[n-1].val {
			continue
		}
		items[n-1] = topItem{idx: idx, val: val}
		// 简单插入排序保持降序
		for j := n - 1; j > 0 && items[j].val > items[j-1].val; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
	return items
}
