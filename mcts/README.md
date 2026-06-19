# MCTS 算法实现

基于 PUCT（Predictor + UCT）的蒙特卡洛树搜索，用于围棋 AI 的走子决策。

---

## 核心类型定义

### Config — 搜索配置

通过 `DefaultConfig()` 获取默认值，按需修改后传入 `NewSearcher`。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `NumSimulations` | `int` | 200 | 每轮搜索的模拟次数，越大棋力越强但越慢 |
| `CPuct` | `float32` | 1.5 | PUCT 探索常数，越大越倾向探索未走过的分支 |
| `SelfPlay` | `bool` | false | 自博弈模式：根节点加 Dirichlet 噪声 + 前 30 手按概率采样 |
| `DirichletAlpha` | `float32` | 0.03 | Dirichlet 浓度参数，越小噪声越集中 |
| `DirichletEps` | `float32` | 0.25 | Dirichlet 噪声混合权重（0.25 = 25% 噪声 + 75% 原始 policy） |
| `PassBonus` | `float32` | 0.0 | Pass 加成：>0 时启用两段式搜索，热门走法全是眼时给 pass 加权 |

**初始化函数：**

```go
func DefaultConfig() Config
```

### Searcher — 搜索器

持有评估器和配置，通过 `NewSearcher` 构造。同一个 searcher 可复用多次调用 `Search`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `eval` | `inference.Evaluator` | CNN 推理接口（不导出） |
| `cfg` | `Config` | 搜索配置（不导出） |

**初始化函数：**

```go
func NewSearcher(eval inference.Evaluator, cfg Config) *Searcher
```

### SearchResult — 搜索结果

`Search()` 的返回值，包含 AI 的最终决策和完整概率分布。

| 字段 | 类型 | 说明 |
|------|------|------|
| `Action` | `int` | 最终走法：0~360 为落子点（`x*19+y`），361 为 pass |
| `VisitProbs` | `[362]float32` | 362 维访问概率向量，和为 1.0，可用于训练神经网络 |
| `RootValue` | `float32` | 根节点平均价值，范围 [-1, 1]，正数 = 当前方优势 |

> `Action` 是从 `VisitProbs` 中选出的单个走法：正常模式取最大概率（argmax），自博弈前 30 手按概率随机采样。

### Node — 树节点（内部类型）

| 字段 | 类型 | 说明 |
|------|------|------|
| `board` | `*board.Board` | 该节点对应的局面（独立深拷贝） |
| `prior` | `float32` | CNN 输出的先验概率（已 mask + 归一化） |
| `children` | `map[int]*Node` | 子节点映射：action → Node |
| `visits` | `int32` | 访问次数 N |
| `valueSum` | `float32` | 价值累积和 W（当前行动方视角） |

**相关常量：**
```go
PassAction = 361   // pass 动作编号
PolicySize = 362   // 策略向量维度（361 落子点 + 1 pass）
```

**相关函数：**
```go
func (n *Node) q() float32              // 返回平均价值 W/N，未访问返回 0
func actionToXY(action int) (x, y int)  // 动作编号 → 棋盘坐标
```

---

## 调用方法

### Search — 执行 MCTS 搜索（唯一对外入口）

```go
func (s *Searcher) Search(ctx context.Context, b *board.Board) (*SearchResult, error)
```

#### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| `ctx` | `context.Context` | 上下文，用于超时控制和取消 |
| `b` | `*board.Board` | 当前棋盘局面（不会被修改，内部用 Clone 深拷贝） |

#### 输出

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `result` | `*SearchResult` | 搜索结果（成功时非 nil） |
| `err` | `error` | 错误信息（CNN 推理失败等），成功时为 nil |

#### 使用示例

```go
// 1. 准备棋盘
b := board.New()
b.Move(3, 3)
b.Move(15, 15)

// 2. 准备评估器（连接 Python 推理服务）
httpClient, _ := inference.NewHTTPClient("http://127.0.0.1:8000/predict", 30*time.Second)
batcher, _ := inference.NewBatcher(httpClient, inference.DefaultBatcherConfig())
defer batcher.Close()

// 3. 配置 MCTS
cfg := mcts.DefaultConfig()
cfg.NumSimulations = 200
// cfg.SelfPlay = true    // 自博弈时开启

// 4. 搜索
searcher := mcts.NewSearcher(batcher, cfg)
result, err := searcher.Search(context.Background(), b)
if err != nil {
    panic(err)
}

// 5. 使用结果
if result.Action == 361 {
    fmt.Println("AI 选择 pass")
} else {
    x, y := result.Action/19, result.Action%19
    fmt.Printf("AI 落子 (%d, %d)\n", x, y)
}
fmt.Printf("局面估值: %.4f\n", result.RootValue)
```

---

## 搜索流程

```
Search(ctx, board)
  │
  ├── 1. 创建根节点，expand() 展开（CNN 推理 → policy + value）
  │
  ├── 2. 终局检查：若已终局，直接返回 pass
  │
  ├── 3. 运行 NumSimulations 次 simulate()：
  │       ├── 选择（Selection）：沿树向下，每层用 PUCT 公式选最优子节点
  │       │       score = Q + U + passBonus
  │       │       Q = -child.q()
  │       │       U = CPuct × prior × √N_parent / (1 + N_child)
  │       │
  │       ├── 扩展（Expansion）：到达叶子 → CNN 推理 → mask → 归一化 → 创建子节点
  │       │       └── 终局时：直接用 terminalValue()（tanh 目数差）
  │       │
  │       └── 回传（Backpropagation）：沿路径反向传播 value（negamax）
  │
  ├── 4. pickMove()：从根节点子节点中选出最终走法
  │       ├── 正常模式：argmax（访问次数最多的）
  │       └── 自博弈前 30 手：按访问概率随机采样
  │
  └── 5.（可选）PassBonus 第二轮：
         若 PassBonus > 0 且前 3 热门非 pass 走法全是己方眼，
         重做整轮搜索并给 pass 加 bonus
```

---

## 随机化机制

| 机制 | 触发条件 | 作用 |
|------|---------|------|
| Dirichlet 噪声 | `SelfPlay=true` | 根节点先验概率混入噪声，增加探索多样性 |
| 概率采样选子 | `SelfPlay=true` + 前 30 手 | 按访问概率随机采样走法（而非取 max） |

---

## 关键公式

**PUCT 选择分数：**

$$score(a) = -Q_{child} + C_{puct} \cdot P(a) \cdot \frac{\sqrt{N_{parent}}}{1 + N_{child}}$$

**终局估值（tanh 压缩）：**

$$v = \tanh\left(\frac{black\\_lead}{15}\right), \quad black\\_lead = 黑目 - (白目 + 7.5)$$

领先 15 目时 value ≈ 0.76，领先 30 目以上接近 ±1。

---

## 文件结构

| 文件 | 说明 |
|------|------|
| `mcts.go` | 搜索主逻辑：Search、simulate、expand、pickMove、随机数工具函数 |
| `node.go` | 树节点定义、动作编码常量、坐标转换 |

