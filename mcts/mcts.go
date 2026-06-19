package mcts

import (
	"context"
	"math"
	"math/rand"
	"sort"

	"go-ai-infer/board"
	"go-ai-infer/inference"
)

const (
	// komi 贴目。board.Score() 不含贴目，这里补上，与 Python go_game 的 komi=7.5 一致。
	komi = 7.5
	// valueScale 终局目数压到 [-1,1] 的缩放，对应 Python score_to_value 的 value_scale。
	valueScale = 15.0
)

// Config MCTS 配置，字段语义对齐 Python select_move_with_policy 的参数。
type Config struct {
	NumSimulations int     // 每轮模拟次数（默认 200）
	CPuct          float32 // PUCT 探索常数（默认 1.5）
	SelfPlay       bool    // 自博弈：根节点加 Dirichlet 噪声，且前 30 手按访问概率采样
	DirichletAlpha float32 // Dirichlet 浓度参数（默认 0.03）
	DirichletEps   float32 // Dirichlet 噪声权重（默认 0.25）

	// PassBonus 对应 Python 的 pass_bonus。
	// > 0 时启用两段式：先正常搜一轮，若访问最高的前 3 个非 pass 走法全是己方眼，
	// 则整轮重搜，并在选择阶段给 pass 的 PUCT 分数加上该 bonus。默认 0（关闭）。
	PassBonus float32
}

// DefaultConfig 返回与 Python 默认参数对齐的配置。
func DefaultConfig() Config {
	return Config{
		NumSimulations: 200,
		CPuct:          1.5,
		SelfPlay:       false,
		DirichletAlpha: 0.03,
		DirichletEps:   0.25,
		PassBonus:      0.0,
	}
}

// SearchResult 搜索结果。
type SearchResult struct {
	Action     int                 // 最终动作：0..360 为落子点，361 为 pass
	VisitProbs [PolicySize]float32 // 362 维：各动作访问次数 / 总访问数（对外返回向量）
	RootValue  float32             // 根节点平均价值
}

// Searcher MCTS 搜索器。
type Searcher struct {
	eval inference.Evaluator
	cfg  Config
}

// NewSearcher 构造搜索器。
func NewSearcher(eval inference.Evaluator, cfg Config) *Searcher {
	return &Searcher{eval: eval, cfg: cfg}
}

// Search 执行 MCTS，对应 Python select_move_with_policy。
// b 是根局面，搜索过程不会修改它（子节点都用 b.Clone() 深拷贝）。
func (s *Searcher) Search(ctx context.Context, b *board.Board) (*SearchResult, error) {
	round := b.Round() // 等价于 Python 的 len(moves)，用于自博弈温度切换

	// ---- 第一轮：正常 MCTS（pass_bonus = 0）----
	root := newRootNode(b)
	rootValue, err := s.expand(ctx, root, s.cfg.SelfPlay)
	if err != nil {
		return nil, err
	}
	// 终局：直接返回 pass。
	if root.board.IsFinish() == 1 {
		return &SearchResult{Action: PassAction, RootValue: rootValue}, nil
	}
	if err := s.runSimulations(ctx, root, 0); err != nil {
		return nil, err
	}
	res := s.pickMove(root, round)

	// ---- 第二轮：若启用 pass_bonus 且前 3 热门非 pass 走法全是眼，重做 ----
	if s.cfg.PassBonus > 0 && topMovesAllEye(root, b, 3) {
		root2 := newRootNode(b)
		if _, err := s.expand(ctx, root2, s.cfg.SelfPlay); err != nil {
			return nil, err
		}
		if err := s.runSimulations(ctx, root2, s.cfg.PassBonus); err != nil {
			return nil, err
		}
		res = s.pickMove(root2, round)
	}

	return res, nil
}

func (s *Searcher) runSimulations(ctx context.Context, root *Node, passBonus float32) error {
	for i := 0; i < s.cfg.NumSimulations; i++ {
		if err := s.simulate(ctx, root, passBonus); err != nil {
			return err
		}
	}
	return nil
}

// simulate 跑一次模拟：选择 -> 扩展 -> 回传，对应 Python _mcts_search 单次循环体。
func (s *Searcher) simulate(ctx context.Context, root *Node, passBonus float32) error {
	node := root
	path := []*Node{node}

	// ---- 选择 ----
	for len(node.children) > 0 {
		total := node.visits // PUCT 分子用父节点自身 visits
		if total < 1 {
			total = 1
		}
		bestAction := -1
		bestScore := float32(math.Inf(-1))

		// 按动作编号升序遍历，复刻 Python dict 插入顺序（避免 Go map 随机序导致的不确定 tie-break）。
		for action := 0; action <= PassAction; action++ {
			child, ok := node.children[action]
			if !ok {
				continue
			}
			q := -child.q() // negamax
			u := s.cfg.CPuct * child.prior * float32(math.Sqrt(float64(total))) / float32(1+child.visits)
			score := q + u
			if action == PassAction {
				score += passBonus // pass 加分
			}
			if score > bestScore {
				bestScore = score
				bestAction = action
			}
		}
		if bestAction == -1 {
			break // 理论上不会发生：pass 恒在
		}
		node = node.children[bestAction]
		if err := node.ensureBoard(); err != nil {
			return err
		}
		path = append(path, node)
	}

	// ---- 扩展 ----
	value, err := s.expand(ctx, node, false)
	if err != nil {
		return err
	}

	// ---- 回传（negamax）----
	for i := len(path) - 1; i >= 0; i-- {
		path[i].visits++
		path[i].valueSum += value
		value = -value
	}
	return nil
}

// expand 扩展节点，返回 value（当前行动方视角，[-1,1]），对应 Python expand。
func (s *Searcher) expand(ctx context.Context, node *Node, addNoise bool) (float32, error) {
	b := node.board

	// ① 终局：不调用 CNN，用简化目数作为 value。
	if b.IsFinish() == 1 {
		return terminalValue(b), nil
	}

	// ② CNN 推理（b.Tensor() 已是 inference.Features，无需转换）。
	eval, err := s.eval.Evaluate(ctx, b.Tensor())
	if err != nil {
		return 0, err
	}

	// 模型 policy 先经过合法手 mask（board.Mask 为 361 维，不含 pass）。
	mask := b.Mask()
	var priors [PolicySize]float32
	legal := make([]int, 0, PolicySize)
	for p := 0; p < board.Points; p++ {
		if mask[p] == 1 {
			priors[p] = eval.Policy[p]
			legal = append(legal, p)
		}
	}
	// 非终局时 pass 恒合法。
	priors[PassAction] = eval.Policy[PassAction]
	legal = append(legal, PassAction)

	if len(legal) == 0 {
		return eval.Value, nil
	}

	// ③ 根节点 Dirichlet 噪声（仅 self_play）。
	if addNoise {
		noise := dirichlet(s.cfg.DirichletAlpha, len(legal))
		for i, a := range legal {
			priors[a] = (1-s.cfg.DirichletEps)*priors[a] + s.cfg.DirichletEps*noise[i]
		}
	}

	// ④ 归一化（仅 legal；和 <= 0 则均匀分布）。
	var sum float32
	for _, a := range legal {
		sum += priors[a]
	}
	if sum <= 0 {
		u := float32(1) / float32(len(legal))
		for _, a := range legal {
			priors[a] = u
		}
	} else {
		for _, a := range legal {
			priors[a] /= sum
		}
	}

	// ⑤ 只创建轻量子节点。棋盘在子节点首次被选中时再 Clone + Move。
	node.children = make(map[int]*Node, len(legal))
	for _, a := range legal {
		node.children[a] = newChildNode(node, a, priors[a])
	}

	return eval.Value, nil
}

// pickMove 从搜索完成的 root 选出最终走法，对应 Python _pick_move。
// round 为根局面手数（= Python len(moves)）。
func (s *Searcher) pickMove(root *Node, round int) *SearchResult {
	var visitProbs [PolicySize]float32
	var total int32
	for _, child := range root.children {
		total += child.visits
	}

	if len(root.children) > 0 && total > 0 {
		for action, child := range root.children {
			visitProbs[action] = float32(child.visits) / float32(total)
		}
	} else if len(root.children) > 0 {
		for action, child := range root.children {
			visitProbs[action] = child.prior
		}
	}

	round0 := round
	var sumProbs float32
	for _, v := range visitProbs {
		sumProbs += v
	}
	var chosen int
	if s.cfg.SelfPlay && round0 < 30 && sumProbs > 0 {
		chosen = sampleByProbs(visitProbs[:], sumProbs)
	} else {
		chosen = argmax(visitProbs[:])
	}

	var rootValue float32
	if len(root.children) > 0 && total > 0 {
		for _, child := range root.children {
			rootValue += (float32(child.visits) / float32(total)) * (-child.q())
		}
	}

	return &SearchResult{Action: chosen, VisitProbs: visitProbs, RootValue: rootValue}
}

// topMovesAllEye 检查访问次数最高的前 topN 个非 pass 走法是否全是己方眼，
// 对应 Python _top_moves_all_eye。注意：眼位用根局面 rootBoard 判断。
func topMovesAllEye(root *Node, rootBoard *board.Board, topN int) bool {
	if len(root.children) == 0 {
		return false
	}
	type mv struct {
		action int
		visits int32
	}
	nonpass := make([]mv, 0, len(root.children))
	for action := 0; action <= PassAction; action++ { // 升序收集，配合稳定排序复刻 Python tie-break
		child, ok := root.children[action]
		if !ok || action == PassAction {
			continue
		}
		nonpass = append(nonpass, mv{action, child.visits})
	}
	if len(nonpass) == 0 {
		return true // 没有非 pass 走法
	}
	sort.SliceStable(nonpass, func(i, j int) bool { return nonpass[i].visits > nonpass[j].visits })

	n := topN
	if len(nonpass) < n {
		n = len(nonpass)
	}
	for i := 0; i < n; i++ {
		x, y := actionToXY(nonpass[i].action)
		if rootBoard.IsEyes(x, y) != 1 {
			return false
		}
	}
	return true
}

// terminalValue 计算终局 value（当前行动方视角），对应 Python expand 终局分支。
func terminalValue(b *board.Board) float32 {
	s := b.Score() // [黑目, 白目, 中立]，不含贴目
	blackLead := float64(s[0]) - (float64(s[1]) + komi)
	v := float32(math.Tanh(blackLead / valueScale)) // = score_to_value(score)
	if b.Round()%2 == 0 {                           // 偶数手轮到黑走
		return v
	}
	return -v
}

// argmax 返回最大值下标（取首个最大值）。
func argmax(v []float32) int {
	best := 0
	bestVal := float32(math.Inf(-1))
	for i, x := range v {
		if x > bestVal {
			bestVal = x
			best = i
		}
	}
	return best
}

// sampleByProbs 按权重（传入其和 sum）采样一个下标。
func sampleByProbs(probs []float32, sum float32) int {
	r := rand.Float32() * sum
	var acc float32
	for i, p := range probs {
		acc += p
		if r <= acc {
			return i
		}
	}
	return len(probs) - 1
}

// dirichlet 采样长度为 n、参数都为 alpha 的 Dirichlet 分布。
func dirichlet(alpha float32, n int) []float32 {
	out := make([]float32, n)
	var sum float64
	a := float64(alpha)
	for i := 0; i < n; i++ {
		g := sampleGamma(a)
		out[i] = float32(g)
		sum += g
	}
	if sum <= 0 {
		u := float32(1) / float32(n)
		for i := range out {
			out[i] = u
		}
		return out
	}
	for i := range out {
		out[i] = out[i] / float32(sum)
	}
	return out
}

// sampleGamma 用 Marsaglia-Tsang 方法采样 Gamma(shape, 1)，支持 shape<1。
func sampleGamma(shape float64) float64 {
	if shape < 1 {
		u := rand.Float64()
		return sampleGamma(shape+1) * math.Pow(u, 1.0/shape)
	}
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		var x, v float64
		for {
			x = rand.NormFloat64()
			v = 1 + c*x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := rand.Float64()
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}
