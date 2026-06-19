# Board 棋盘类

Board 类实现了围棋棋盘的核心逻辑，包括落子、提子、禁着判定（含打劫/全局同型再现）、数目、终局判断、眼位判断等功能，并支持将当前局面编码为神经网络的输入张量。

---

# 对外接口（Public API）

以下常量、类型、构造函数和方法供其他 package 使用。

## 常量

```go
const (
    Size   = 19    // 棋盘大小（19 路）
    Points = 361   // 棋盘总点数（19 × 19）

    Empty = 0      // 空位
    Black = 1      // 黑子
    White = 2      // 白子
)
```

其中 `Size` 和 `Points` 引用自 `inference` 包（`inference.BoardSize`、`inference.BoardPoints`），保证与推理服务一致。

## 类型别名

```go
type Board = board
```

`Board` 是 `board` 的公开别名。`board` 为小写，包外不可直接访问其字段；通过 `Board` 别名和公开方法使用。

## 构造函数

### New

```go
func New() *Board
```

创建一个新的空棋盘。`round` 和 `hash` 初始为 0，`blackHashes` 和 `whiteHashes` 已初始化。`boards` 和 `passes` 为零值（空盘面、无 pass）。

---

## 公开方法

### Round — 获取当前回合数

```go
func (cur *board) Round() int
```

返回当前 `round` 值。

- `round = 0`：空棋局
- `round > 0`：奇数表示黑刚走过，偶数表示白刚走过

**不修改原 board。**

---

### Mask — 查看合法落点

```go
func (cur *board) Mask() [Points]int
```

返回一个长度为 361 的向量，第 `p`（`p = x * 19 + y`）位取值为：

- `1`：当前行动方可以落子在该位置
- `0`：不可以（被占据 / 禁着点 / 打劫 / 全局同型再现等）

终局状态下返回全 0。

Mask 不包含 pass，pass 由 MCTS 层自行处理。

**不修改原 board。** 内部调用 `analyzeBoard` 做一次性盘面分析，不对 `cur` 做任何写操作。

---

### Move — 落子

```go
func (cur *board) Move(x int, y int) int
```

坐标范围为 `0 <= x, y <= 18`。特别地，`x = y = -1` 表示 pass。

**返回值：**

| 返回值 | 含义 |
|--------|------|
| `0` | 落子 / pass 合法，board 状态已更新 |
| `1` | 不合法，board 状态不变（原子操作） |

**不合法的情况：**

1. 已终局（连续两手 pass 后）—— 任何落子含 pass 均返回 1
2. 坐标越界（不在 `[0, Size)` 且不是 `(-1, -1)`）
3. 目标点非空（已有棋子）
4. 禁着点（自杀）：落子后己方棋块无气且不能提掉对方棋子
5. 打劫 / 全局同型再现：新产生的局面会使对方重复面临历史上曾出现的局面

**合法时执行的操作：**

1. 在目标点放置棋子
2. 提掉气数为 0 的对方棋块
3. 推进 `boards` 和 `passes` 历史
4. `round` 加一
5. 若为实际落子（非 pass），将新局面哈希插入 `blackHashes` 或 `whiteHashes`

---

### Tensor — 转换为推理输入

```go
func (cur *board) Tensor() inference.Features
```

将当前棋局转换为可直接传递给推理服务的 `inference.Features`（长度为 `FeatureSize = 9 × 361` 的 `float32` 数组）。

**九个通道定义：**

| 通道索引 | 内容 |
|---------|------|
| 0 | `boards[0]` 黑棋位置 |
| 1 | `boards[0]` 白棋位置 |
| 2 | `boards[1]` 黑棋位置 |
| 3 | `boards[1]` 白棋位置 |
| 4 | `boards[2]` 黑棋位置 |
| 5 | `boards[2]` 白棋位置 |
| 6 | `boards[3]` 黑棋位置 |
| 7 | `boards[3]` 白棋位置 |
| 8 | 当前行动方平面：黑走时整张全 1，白走时整张全 0 |

**不修改原 board。**

---

### Score — 数目数

```go
func (cur *board) Score() [3]int
```

返回 `[黑方目数, 白方目数, 中立目数]`。三者之和恒为 `Points = 361`。

**目数归属规则：**

1. 非空位：直接归属对应颜色的棋子
2. 空位连通块（BFS 找出）：统计该连通块相邻的非空棋子颜色集合
   - 只包含黑色 → 全归黑
   - 只包含白色 → 全归白
   - 集合为空，或同时包含黑白 → 归中立

**不修改原 board。**

---

### IsFinish — 判断终局

```go
func (cur *board) IsFinish() int
```

返回值：

- `1`：已终局（倒数两手均为 pass）
- `0`：未终局

**不修改原 board。**

---

### IsEyes — 判断眼

```go
func (cur *board) IsEyes(x int, y int) int
```

判断坐标 `(x, y)`（范围 `0 <= x, y <= 18`）的空位是否是当前行动方的眼。

**眼的定义：** 如果对方不能合法落子在该位置（禁着点），同时该位置周围存在己方棋子，则为眼。

返回值：

- `1`：是眼
- `0`：不是眼（越界、非空、周围无己方棋子、或对方可下）

**不修改原 board。**

---

# 内部实现（Internal）

以下结构体、变量和函数仅供 `board` 包内部使用，包外不可访问。

## 内部结构体

### board — 棋盘

```go
type board struct {
    round       int                        // 当前局面最后一手是第几手
    boards      [4][Points]int             // 倒数四手盘面
    passes      [2]bool                    // 倒数两手是否 pass
    blackHashes map[uint64]bool            // 轮到黑走时历史上出现过的局面哈希
    whiteHashes map[uint64]bool            // 轮到白走时历史上出现过的局面哈希
    hash        uint64                     // 当前局面 Zobrist 哈希
}
```

字段说明详见上文「对外接口」中各方法的描述。关键设计：

- `blackHashes` 和 `whiteHashes` 分表存储：黑走完轮到白时，新局面与 `whiteHashes` 比对；反之与 `blackHashes` 比对
- 初始空盘面不插入哈希表（允许开局连续两次 pass 终局）
- pass 产生的局面不插入哈希表（pass 不参与历史重复判定）

### group — 棋块

```go
type group struct {
    color      int      // 棋块颜色（Black / White）
    stones     []int    // 棋块包含的棋子（线性索引）
    libertyCnt int      // 气数
    hash       uint64   // 棋块 Zobrist 哈希
}
```

一次 BFS 连通块搜索的结果。`hash` 为棋块内各棋子 `zobrist[p][color]` 的异或值。

### positionAnalysis — 盘面分析结果

```go
type positionAnalysis struct {
    gid    [Points]int  // gid[p] 表示点 p 所属棋块编号，空点为 -1
    groups []group      // 当前盘面所有棋块
    hash   uint64       // 整盘棋的 Zobrist 哈希
}
```

由 `analyzeBoard` 一次性生成，一次 O(N) 分析供多次 O(1) 合法性查询复用。`hash` 为所有 `group.hash` 的异或值。

### legalResult — 单点合法性判断结果

```go
type legalResult struct {
    newHash       uint64    // 落子并提子后的新局面哈希
    captured      [4]int    // 被提棋块编号（最多 4 个）
    capturedCount int       // 实际被提棋块数量
}
```

`legalAtWithAnalysis` 的返回值。提子后棋块编号会失效，所以只记录编号而非棋块本身。

---

## 内部变量

### zobrist — Zobrist 哈希表

```go
var zobrist = initZobrist()
```

包级变量，`[Points][3]uint64` 类型。`zobrist[p][color]` 表示在点 `p` 放置 `color` 颜色棋子对应的哈希增量。在包初始化时通过 `initZobrist()` 自动生成。

---

## 内部函数

### initZobrist — 初始化 Zobrist 表

```go
func initZobrist() [Points][3]uint64
```

使用 SplitMix64 伪随机数生成器，为每个点的每种颜色生成独立的 64 位随机数。每个点生成完黑、白两个值后，种子再推进一次，保证三态（空/黑/白）独立。

### splitmix64 — 伪随机数生成

```go
func splitmix64(x uint64) uint64
```

标准 SplitMix64 算法的单步迭代，用于 Zobrist 哈希表的确定性伪随机数生成。

### currentColor — 当前行棋方

```go
func currentColor(round int) int
```

- `round` 为偶数（含 0）→ 返回 `Black`（黑先行）
- `round` 为奇数 → 返回 `White`

### opponent — 对手颜色

```go
func opponent(color int) int
```

`Black` → `White`；`White` → `Black`。

### inRange — 坐标范围检查

```go
func inRange(x int, y int) bool
```

判断 `(x, y)` 是否在 `[0, Size)` 范围内。

### idx — 坐标转线性索引

```go
func idx(x int, y int) int
```

二维坐标转一维索引：`p = x * Size + y`。逆运算为 `x = p / Size`，`y = p % Size`。

### neighbors — 获取邻居

```go
func neighbors(p int, out *[4]int) int
```

获取点 `p` 的上下左右邻居，写入 `out` 数组，返回邻居数量（角点 2，边点 3，内部 4）。

### hasSmall — 小数组查找

```go
func hasSmall(arr *[4]int, cnt int, x int) bool
```

判断 `arr` 的前 `cnt` 个元素中是否包含 `x`。用于 `legalAtWithAnalysis` 中检查某棋块是否已在被提列表中。

### ensureMaps — 确保哈希表已初始化

```go
func (cur *board) ensureMaps()
```

防御性检查：若 `blackHashes` 或 `whiteHashes` 为 nil，则初始化。用于兼容不经过 `New()` 创建的 `board`。

### analyzeBoard — 盘面分析

```go
func analyzeBoard(b [Points]int) positionAnalysis
```

对给定盘面做一次完整 BFS 分析，返回 `positionAnalysis`。这是棋盘分析的核心函数：

- 遍历所有点，遇到未访问的非空点则启动 BFS
- BFS 过程中记录棋块所有棋子、统计气数、计算棋块哈希
- 时间复杂度 O(N)，空间复杂度 O(N)

### legalAtWithAnalysis — 单点合法性判断

```go
func (cur *board) legalAtWithAnalysis(
    p int,
    color int,
    a *positionAnalysis,
    checkRepeat bool,
) (bool, legalResult)
```

基于已有的 `positionAnalysis`，判断 `color` 方能否落在点 `p` 上。`checkRepeat` 控制是否检查打劫/全局同型再现。

要点：

1. 只检查 `p` 周围最多 4 个邻居
2. 提子通过相邻对方棋块 `libertyCnt == 1` 判断
3. 自杀通过「有直接空邻居 / 连接己方多气棋块 / 能提子」判断
4. 新哈希通过棋块哈希异或增量得到，不复制棋盘
5. 单点复杂度 O(1)

### pushBoard — 推进盘面历史

```go
func (cur *board) pushBoard(next [Points]int, isPass bool)
```

将 `next` 设为 `boards[0]`，旧盘面整体后移一位（`boards[0]→[1]→[2]→[3]`，`[3]` 丢弃）。`passes` 同理后移。

---

## Zobrist 哈希原理

Board 使用 Zobrist 哈希实现棋盘局面的快速比较和增量更新：

- **增量更新**：新局面 hash = 原 hash ⊕ 落子点 hash ⊕ 被提棋块 hash。XOR 运算使放子（XOR 一次）和提子（再 XOR 一次）都是 O(1)
- **免复制棋盘**：`legalAtWithAnalysis` 通过哈希增量计算新局面哈希，无需真正模拟落子和提子，即可判断打劫
- **棋块哈希**：棋块内各棋子 `zobrist` 值的异或。提掉整个棋块时，XOR 棋块哈希即可一次性清除所有棋子

---

# 测试与验证

## board_test.go — 单元测试

`board_test.go` 对 board 包的各个函数进行了系统的单元测试，覆盖 13 个测试分组：

| 分组 | 测试函数 | 验证内容 |
|------|---------|---------|
| 1. 基础工具函数 | `TestCurrentColor`、`TestOpponent`、`TestInRange`、`TestIdx`、`TestNeighbors`、`TestHasSmall`、`TestSplitmix64` | 行棋方轮转、对手颜色、坐标范围、索引换算、邻居计算、小数组查找、哈希确定性 |
| 2. analyzeBoard | `TestAnalyzeBoard_Empty` / `_SingleStone` / `_CornerStone` / `_ConnectedGroup` / `_SeparateGroups` / `_HashConsistency` | 空盘、单子、角点（2 气）、连通棋块（并为一组）、分离棋块（各自成组）、哈希一致性 |
| 3. legalAtWithAnalysis | `TestLegalAtWithAnalysis_EmptyPoint` / `_OccupiedPoint` / `_OutOfBounds` / `_Capture` / `_Suicide` / `_Ko` | 空点可下、占点不可下、越界不可下、提子合法、自杀非法、打劫非法 |
| 4. move | `TestMove_Pass` / `_LegalMove` / `_OutOfBounds` / `_OccupiedPoint` / `_CaptureSequence` / `_GameEndDenial` / `_HashRecorded` | pass 后 round+1、合法落子、越界拒绝、占点拒绝、提子序列、终局后拒绝、哈希记录 |
| 5. score | `TestScore_EmptyBoard` / `_AfterMoves` / `_AfterCapture` / `_TerritoryOwnership` / `_TotalAlways361` | 空盘全中立、落子计数、提子后计数、领土归属、总和恒 361 |
| 6. is_finish | `TestIsFinish_Initial` / `_OnePass` / `_TwoPasses` / `_PassMovePass` | 初始未终局、一次 pass 未终局、两次 pass 终局、落子打断 pass 链 |
| 7. is_eyes | `TestIsEyes_RealEye` / `_NotEye_EmptyBoard` / `_NotEye_OpponentTerritory` / `_OutOfBounds` / `_NotEmpty` | 真眼识别、空盘无眼、对方地盘误判修复验证、越界、非空点 |
| 8. tensor | `TestTensor_Dimensions` / `_CurrentPlayerPlane_Black` / `_CurrentPlayerPlane_White` / `_StoneEncoding` / `_HistoryPlanes` | 维度正确、黑方平面全 1、白方平面全 0、棋子编码、历史盘面通道 |
| 9. mask | `TestMask_EmptyBoard` / `_AfterGameEnd` / `_OccupiedPointsNotInMask` | 空盘全合法、终局全 0、占点不在 mask |
| 10. pushBoard | `TestPushBoard_HistoryShift` / `_PassShift` | 盘面历史正确移位、pass 历史正确移位 |
| 11. New / ensureMaps | `TestNew` / `TestEnsureMaps` | New 初始化完整、ensureMaps 防御性初始化 |
| 12. Zobrist | `TestZobrist_Initialization` | 哈希值非零且黑白不同 |
| 13. 集成测试 | `TestIntegration_FullGame` | 从空盘到终局的完整对局流程：落子 → score → mask → pass → 终局拒绝 → tensor |

测试的设计思路是对每个函数按正常情况、边界情况、异常情况分别构造用例，并通过 `newBoardWithStones` 辅助函数直接构造特定盘面，避免依赖 `Move` 来搭建测试场景（保证测试的独立性）。

---

## board_interaction.go — 交互式对局验证

`board_interaction.go` 提供了一个双方轮流下棋的终端交互程序，用于人工验证棋盘逻辑的正确性。

**原理：**

- 创建空棋盘，循环读取用户输入
- 支持三种输入：落子坐标（`x y`）、停着（`pass` 或 `-1 -1`）、退出（`quit`）
- 每手棋调用 `Move` 执行，非法落子会提示错误并允许重试
- 每次落子后调用 `printBoard` 打印当前盘面（● 黑 / ○ 白 / · 空）
- 终局后自动调用 `Score` 输出目数结果

## 测试命令

```bash
# 运行交互式测试
go test -v -run TestInteractivePlay -timeout 0

# 运行 board 包的所有测试（带详细输出）
go test ./board/ -v

# 只运行某个特定测试
go test ./board/ -v -run TestMove

# 运行并显示覆盖率
go test ./board/ -v -cover

# 如果在 board 目录内部，可以省略 ./board/
cd board && go test -v
```